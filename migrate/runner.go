package migrate

// runner.go — Core migration runner.
// Implements schema_migration_script and record_migration_script execution,
// shadow table strategy, batch sizing, resume gate, and FK ordering.
// Design: bakeoff#27 (Phase 1 sign-off 050558ZJUN26, Phase 2 spec).

import (
	"context"
	"fmt"
	"log"
	"math"
	"math/rand"
	"strings"
	"time"

	"github.com/d5/tengo/v2"
	"github.com/jackc/pgx/v5"
)

// Config holds runtime options for the migration runner.
type Config struct {
	// BatchSize: >0 = fixed; 0 = dynamic (default); -1 = invalid.
	BatchSize int
	// MaxRejects: 0 = fail on first; -1 = collect all; default 10.
	MaxRejects int
	// IgnoreIsFatal: treat any ignoreRecord as a fatal error.
	IgnoreIsFatal bool
	// DryRun: execute scripts, accumulate counts, report without committing.
	DryRun bool
	// ProcessingTimeTarget: target processing time per batch (default 20s).
	ProcessingTimeTarget float64
	// CommitTimeTarget: target commit time per batch (default 1s).
	CommitTimeTarget float64
}

// DefaultConfig returns a Config with production-safe defaults.
func DefaultConfig() Config {
	return Config{
		BatchSize:            0,
		MaxRejects:           10,
		IgnoreIsFatal:        false,
		DryRun:               false,
		ProcessingTimeTarget: 20.0,
		CommitTimeTarget:     1.0,
	}
}

// MigrationRunner executes schema and record migration scripts against a PostgreSQL DB.
type MigrationRunner struct {
	conn *pgx.Conn
	cfg  Config
}

// NewRunner constructs a MigrationRunner with the provided pgx connection and config.
func NewRunner(conn *pgx.Conn, cfg Config) *MigrationRunner {
	return &MigrationRunner{conn: conn, cfg: cfg}
}

// SchemaVersion mirrors the schema_versions table row.
type SchemaVersion struct {
	ID                    int
	Description           string
	AllowMigration        bool
	SchemaMigrationScript *string
	RecordMigrationScript *string
}

// SchemaTable mirrors the schema_tables table row.
type SchemaTable struct {
	TableID          string
	TableName        string
	UUIDNamespace    string
	CurrentVersionID int
	DeprecatedAt     *time.Time
}

// Run executes the migration for the given schema_version_id.
// This is the main entry point called by the CLI.
func (r *MigrationRunner) Run(ctx context.Context, schemaVersionID int) error {
	if r.cfg.BatchSize < 0 {
		return fmt.Errorf("--batch-size -1 is invalid; use 0 for dynamic sizing or N>0 for fixed")
	}

	// 1. Resume gate — detect orphan shadow tables.
	if err := r.resumeGate(ctx); err != nil {
		return fmt.Errorf("resume gate: %w", err)
	}

	// 2. Load schema version.
	sv, err := r.loadSchemaVersion(ctx, schemaVersionID)
	if err != nil {
		return fmt.Errorf("load schema version %d: %w", schemaVersionID, err)
	}
	if !sv.AllowMigration {
		return fmt.Errorf("schema version %d has allow_migration=false; set to true to proceed", schemaVersionID)
	}

	// 3. Load all schema_tables and build FK migration order.
	tables, err := r.loadSchemaTables(ctx)
	if err != nil {
		return fmt.Errorf("load schema tables: %w", err)
	}
	orderedTables, err := r.topologicalOrder(ctx, tables)
	if err != nil {
		return fmt.Errorf("topological sort: %w", err)
	}

	// 4. Execute schema_migration_script (DDL phase).
	if sv.SchemaMigrationScript != nil && *sv.SchemaMigrationScript != "" {
		log.Printf("[migrate] running schema_migration_script for version %d", schemaVersionID)
		if err := r.runSchemaScript(ctx, *sv.SchemaMigrationScript, sv); err != nil {
			return fmt.Errorf("schema_migration_script: %w", err)
		}
	}

	// 5. Execute record_migration_script (per-record phase).
	if sv.RecordMigrationScript != nil && *sv.RecordMigrationScript != "" {
		log.Printf("[migrate] running record_migration_script for version %d across %d tables", schemaVersionID, len(orderedTables))
		for _, table := range orderedTables {
			if table.DeprecatedAt != nil {
				// Deprecated table — skip (handled by schema_tables_join edges).
				continue
			}
			if err := r.runRecordMigration(ctx, sv, table); err != nil {
				return fmt.Errorf("record migration for table %q: %w", table.TableName, err)
			}
		}
	}

	log.Printf("[migrate] version %d migration complete", schemaVersionID)
	return nil
}

// runSchemaScript compiles and executes a Tengo DDL script.
// Script errors are fatal (abort migration entirely).
func (r *MigrationRunner) runSchemaScript(ctx context.Context, script string, sv SchemaVersion) error {
	builtins := schemaBuiltins(ctx, r.conn)

	compiled, err := compileScript(script, builtins)
	if err != nil {
		return fmt.Errorf("script compile error (fatal): %w", err)
	}

	if r.cfg.DryRun {
		log.Printf("[migrate:dry-run] schema_migration_script would execute (skipped)")
		return nil
	}

	if err := compiled.Run(); err != nil {
		return fmt.Errorf("script runtime error (fatal): %w", err)
	}
	return nil
}

// runRecordMigration runs the record_migration_script for a single table.
// Uses shadow table strategy unless inline path is taken.
func (r *MigrationRunner) runRecordMigration(ctx context.Context, sv SchemaVersion, table SchemaTable) error {
	script := *sv.RecordMigrationScript

	// Pre-compile once; parse errors are fatal.
	builtins := recordBuiltins(ctx, r.conn, &Record{}) // placeholder record for compile check
	_, err := compileScript(script, builtins)
	if err != nil {
		return fmt.Errorf("record script compile error (fatal): %w", err)
	}

	// Determine shadow table name.
	shadowTable := fmt.Sprintf("_bakeoff_migration_%s_%d", sanitizeName(table.TableName), sv.ID)

	// Inline path: if no structural DDL change and UUID namespace unchanged,
	// migrate against the live table.
	// TODO(phase2b): implement inline detection via schema fingerprint comparison.
	// For now always use the shadow table path.
	inlinePath := false

	if inlinePath {
		return r.runRecordMigrationInline(ctx, sv, table, script)
	}

	return r.runRecordMigrationShadow(ctx, sv, table, script, shadowTable)
}

// runRecordMigrationShadow implements the shadow table migration strategy.
// 1. Create shadow table with destination DDL.
// 2. Migrate records into shadow table.
// 3. Atomic rename: old → _bakeoff_old_<table>, shadow → <table>.
func (r *MigrationRunner) runRecordMigrationShadow(
	ctx context.Context,
	sv SchemaVersion,
	table SchemaTable,
	script string,
	shadowTable string,
) error {
	stateTable := "_bakeoff_migration_state"

	// Ensure state tracking table exists.
	if err := r.ensureMigrationStateTable(ctx, stateTable); err != nil {
		return err
	}

	// Check for existing checkpoint (resume from prior run).
	lastRowID, err := r.loadCheckpoint(ctx, stateTable, table.TableName, sv.ID)
	if err != nil {
		return err
	}

	if lastRowID == 0 {
		// Fresh start: create shadow table (copy DDL from source).
		log.Printf("[migrate] creating shadow table %s for %s", shadowTable, table.TableName)
		if !r.cfg.DryRun {
			if err := r.createShadowTable(ctx, table.TableName, shadowTable); err != nil {
				return err
			}
			if err := r.saveCheckpoint(ctx, stateTable, table.TableName, sv.ID, 0); err != nil {
				return err
			}
		}
	} else {
		log.Printf("[migrate] resuming %s from row id %d", table.TableName, lastRowID)
	}

	// Run migration loop.
	stats, err := r.migrateRecords(ctx, sv, table, script, shadowTable, stateTable, lastRowID)
	if err != nil {
		return err
	}

	log.Printf("[migrate] %s: %d migrated, %d rejected, %d ignored",
		table.TableName, stats.migrated, stats.rejected, stats.ignored)

	// Atomic swap: only when all records complete successfully.
	if !r.cfg.DryRun && stats.rejected == 0 {
		oldName := fmt.Sprintf("_bakeoff_old_%s", sanitizeName(table.TableName))
		log.Printf("[migrate] swapping %s → %s, %s → %s", table.TableName, oldName, shadowTable, table.TableName)
		if err := r.atomicSwap(ctx, table.TableName, shadowTable, oldName); err != nil {
			return fmt.Errorf("atomic swap: %w", err)
		}
		// Drop state table now that this migration is complete.
		if err := r.dropCheckpoint(ctx, stateTable, table.TableName, sv.ID); err != nil {
			return err
		}
		// Drop state table entirely if empty.
		r.maybeDropStateTable(ctx, stateTable)
	} else if stats.rejected > 0 {
		return fmt.Errorf("%d records rejected during migration of %s (shadow table preserved for inspection)",
			stats.rejected, table.TableName)
	}

	return nil
}

// runRecordMigrationInline migrates records against the live table (no shadow).
func (r *MigrationRunner) runRecordMigrationInline(
	ctx context.Context,
	sv SchemaVersion,
	table SchemaTable,
	script string,
) error {
	// TODO(phase2b): implement inline record migration.
	return fmt.Errorf("inline migration path not yet implemented for table %q", table.TableName)
}

// migrationStats tracks per-run counters.
type migrationStats struct {
	migrated int
	rejected int
	ignored  int
}

// migrateRecords runs the record migration loop with dynamic batch sizing.
func (r *MigrationRunner) migrateRecords(
	ctx context.Context,
	sv SchemaVersion,
	table SchemaTable,
	script, shadowTable, stateTable string,
	resumeFromID int64,
) (migrationStats, error) {
	var stats migrationStats

	// Dynamic batch state — 6 variables per bakeoff#27 (050558ZJUN26).
	var (
		totalRecords        int
		lastBatchSize       int
		totalProcessingTime float64
		totalCommitTime     float64
		lastProcessingTime  float64
		lastCommitTime      float64
	)

	currentBatchSize := 64 // Batch 0 always 64 records.
	if r.cfg.BatchSize > 0 {
		currentBatchSize = r.cfg.BatchSize
	}

	var rejects []rejectEntry
	warnedTimeLimitOnce := false
	firstRejection := false
	lastID := resumeFromID

	for {
		// Fetch a batch from the source table, ordered by primary key.
		rows, err := r.fetchBatch(ctx, table.TableName, lastID, currentBatchSize)
		if err != nil {
			return stats, fmt.Errorf("fetchBatch: %w", err)
		}
		if len(rows) == 0 {
			break // All records processed.
		}

		// Randomize order after first rejection (bakeoff#27 spec).
		if firstRejection {
			rand.Shuffle(len(rows), func(i, j int) { rows[i], rows[j] = rows[j], rows[i] })
		}

		processStart := time.Now()

		batchMigrated := 0
		for _, row := range rows {
			rec := NewRecord(row)
			// Build per-record builtins with this record's context.
			recBuiltins := recordBuiltins(ctx, r.conn, rec)
			compiled, err := compileScript(script, recBuiltins)
			if err != nil {
				// Script compile error is fatal.
				return stats, fmt.Errorf("record script compile error (fatal): %w", err)
			}

			// Inject `this` variable.
			if err := compiled.Set("this", rec.toTengoMap()); err != nil {
				return stats, fmt.Errorf("script set this: %w", err)
			}

			if !r.cfg.DryRun {
				if err := compiled.Run(); err != nil {
					// Runtime template error counts as implicit reject.
					rejects = append(rejects, rejectEntry{
						RowID:  primaryKeyOf(row),
						Table:  table.TableName,
						Reason: fmt.Sprintf("script runtime error: %v", err),
					})
					stats.rejected++
					if !firstRejection {
						firstRejection = true
					}
					if r.cfg.MaxRejects >= 0 && stats.rejected > r.cfg.MaxRejects {
						return stats, fmt.Errorf("max-rejects (%d) exceeded; aborting", r.cfg.MaxRejects)
					}
					continue
				}
			}

			if rec.rejected {
				rejects = append(rejects, rejectEntry{
					RowID:  primaryKeyOf(row),
					Table:  table.TableName,
					Reason: rec.rejectMsg,
				})
				stats.rejected++
				if !firstRejection {
					firstRejection = true
				}
				if r.cfg.MaxRejects >= 0 && stats.rejected > r.cfg.MaxRejects {
					return stats, fmt.Errorf("max-rejects (%d) exceeded; aborting", r.cfg.MaxRejects)
				}
				continue
			}

			if rec.ignored {
				log.Printf("[migrate] ignored record in %s: %s", table.TableName, rec.ignoreMsg)
				stats.ignored++
				if r.cfg.IgnoreIsFatal {
					return stats, fmt.Errorf("--ignore-is-fatal: record ignored in %s: %s", table.TableName, rec.ignoreMsg)
				}
				continue
			}

			if !r.cfg.DryRun {
				if err := r.writeRecordToShadow(ctx, shadowTable, rec.MergedRow()); err != nil {
					return stats, fmt.Errorf("write to shadow table: %w", err)
				}
			}
			batchMigrated++
		}

		processingDur := time.Since(processStart).Seconds()

		// Update last seen row ID for checkpointing.
		if len(rows) > 0 {
			if id, ok := rows[len(rows)-1]["id"]; ok {
				if idInt, ok := id.(int64); ok {
					lastID = idInt
				}
			}
		}

		// Commit batch.
		commitStart := time.Now()
		if !r.cfg.DryRun {
			if err := r.saveCheckpoint(ctx, stateTable, table.TableName, sv.ID, lastID); err != nil {
				return stats, err
			}
		}
		commitDur := time.Since(commitStart).Seconds()

		stats.migrated += batchMigrated

		// Update dynamic batch state.
		totalRecords += len(rows)
		lastBatchSize = len(rows)
		totalProcessingTime += processingDur
		totalCommitTime += commitDur
		lastProcessingTime = processingDur
		lastCommitTime = commitDur

		// Check time limits and warn once.
		if !warnedTimeLimitOnce {
			if processingDur > r.cfg.ProcessingTimeTarget || commitDur > r.cfg.CommitTimeTarget {
				log.Printf("[migrate] WARNING: batch time exceeded targets (processing=%.2fs target=%.2fs, commit=%.2fs target=%.2fs)",
					processingDur, r.cfg.ProcessingTimeTarget, commitDur, r.cfg.CommitTimeTarget)
				warnedTimeLimitOnce = true
			}
		}

		// Compute next batch size if dynamic.
		if r.cfg.BatchSize == 0 && totalRecords > 0 && len(rows) == currentBatchSize {
			// After batch 0 (or when we have data), compute.
			if lastCommitTime > 0 && lastProcessingTime > 0 {
				nextByProcessing := r.cfg.ProcessingTimeTarget *
					(3*(float64(totalRecords)/totalProcessingTime)+(float64(lastBatchSize)/lastProcessingTime)) / 4
				nextByCommit := r.cfg.CommitTimeTarget *
					(3*(float64(totalRecords)/totalCommitTime)+(float64(lastBatchSize)/lastCommitTime)) / 4
				next := math.Min(nextByProcessing, nextByCommit)
				if next < 1 {
					next = 1
				}
				currentBatchSize = int(math.Round(next))
				log.Printf("[migrate] dynamic batch: next=%d (by_proc=%.0f, by_commit=%.0f)",
					currentBatchSize, nextByProcessing, nextByCommit)
			}
		}

		if len(rows) < currentBatchSize {
			break // Last batch was smaller than requested — we're done.
		}
	}

	// Report all rejects at the end.
	if len(rejects) > 0 {
		log.Printf("[migrate] REJECTS for %s:", table.TableName)
		for _, re := range rejects {
			log.Printf("  row=%v reason=%s", re.RowID, re.Reason)
		}
	}

	return stats, nil
}

// rejectEntry records a rejected record for end-of-run reporting.
type rejectEntry struct {
	RowID any
	Table string
	Reason string
}

// fetchBatch fetches up to batchSize rows from table where id > afterID.
func (r *MigrationRunner) fetchBatch(ctx context.Context, table string, afterID int64, batchSize int) ([]map[string]any, error) {
	sql := fmt.Sprintf("SELECT * FROM %s WHERE id > $1 ORDER BY id ASC LIMIT $2", table)
	rows, err := r.conn.Query(ctx, sql, afterID, batchSize)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return pgx.CollectRows(rows, func(row pgx.CollectableRow) (map[string]any, error) {
		return pgx.RowToMap(row)
	})
}

// writeRecordToShadow inserts a migrated record into the shadow table.
// Uses a dynamic INSERT built from the row's keys.
func (r *MigrationRunner) writeRecordToShadow(ctx context.Context, shadowTable string, row map[string]any) error {
	if len(row) == 0 {
		return nil
	}
	cols := make([]string, 0, len(row))
	vals := make([]any, 0, len(row))
	for k, v := range row {
		cols = append(cols, k)
		vals = append(vals, v)
	}
	placeholders := make([]string, len(cols))
	for i := range cols {
		placeholders[i] = fmt.Sprintf("$%d", i+1)
	}
	sql := fmt.Sprintf("INSERT INTO %s (%s) VALUES (%s) ON CONFLICT DO NOTHING",
		shadowTable,
		strings.Join(cols, ", "),
		strings.Join(placeholders, ", "))
	_, err := r.conn.Exec(ctx, sql, vals...)
	return err
}

// createShadowTable creates a shadow table with the same DDL as the source table.
func (r *MigrationRunner) createShadowTable(ctx context.Context, sourceTable, shadowTable string) error {
	// Use CREATE TABLE ... LIKE to clone structure without data.
	sql := fmt.Sprintf("CREATE TABLE IF NOT EXISTS %s (LIKE %s INCLUDING ALL)", shadowTable, sourceTable)
	_, err := r.conn.Exec(ctx, sql)
	return err
}

// atomicSwap renames: source → oldName, shadow → source.
// Uses two sequential renames in the same transaction.
func (r *MigrationRunner) atomicSwap(ctx context.Context, source, shadow, oldName string) error {
	tx, err := r.conn.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx) //nolint:errcheck
	if _, err := tx.Exec(ctx, fmt.Sprintf("ALTER TABLE %s RENAME TO %s", source, oldName)); err != nil {
		return err
	}
	if _, err := tx.Exec(ctx, fmt.Sprintf("ALTER TABLE %s RENAME TO %s", shadow, source)); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

// ensureMigrationStateTable creates _bakeoff_migration_state if it doesn't exist.
func (r *MigrationRunner) ensureMigrationStateTable(ctx context.Context, stateTable string) error {
	sql := fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
		table_name      TEXT    NOT NULL,
		version_id      INTEGER NOT NULL,
		last_row_id     BIGINT  NOT NULL DEFAULT 0,
		updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
		PRIMARY KEY (table_name, version_id)
	)`, stateTable)
	_, err := r.conn.Exec(ctx, sql)
	return err
}

// loadCheckpoint returns the last_row_id for (tableName, versionID), or 0 if none.
func (r *MigrationRunner) loadCheckpoint(ctx context.Context, stateTable, tableName string, versionID int) (int64, error) {
	var lastID int64
	err := r.conn.QueryRow(ctx,
		fmt.Sprintf("SELECT last_row_id FROM %s WHERE table_name=$1 AND version_id=$2", stateTable),
		tableName, versionID).Scan(&lastID)
	if err == pgx.ErrNoRows {
		return 0, nil
	}
	if err != nil {
		return 0, err
	}
	return lastID, nil
}

// saveCheckpoint upserts the checkpoint for (tableName, versionID).
func (r *MigrationRunner) saveCheckpoint(ctx context.Context, stateTable, tableName string, versionID int, lastID int64) error {
	sql := fmt.Sprintf(`INSERT INTO %s (table_name, version_id, last_row_id, updated_at)
		VALUES ($1, $2, $3, now())
		ON CONFLICT (table_name, version_id) DO UPDATE
		SET last_row_id=EXCLUDED.last_row_id, updated_at=EXCLUDED.updated_at`, stateTable)
	_, err := r.conn.Exec(ctx, sql, tableName, versionID, lastID)
	return err
}

// dropCheckpoint removes the checkpoint row for (tableName, versionID).
func (r *MigrationRunner) dropCheckpoint(ctx context.Context, stateTable, tableName string, versionID int) error {
	_, err := r.conn.Exec(ctx,
		fmt.Sprintf("DELETE FROM %s WHERE table_name=$1 AND version_id=$2", stateTable),
		tableName, versionID)
	return err
}

// maybeDropStateTable drops _bakeoff_migration_state if it has no rows left.
func (r *MigrationRunner) maybeDropStateTable(ctx context.Context, stateTable string) {
	var count int
	if err := r.conn.QueryRow(ctx, fmt.Sprintf("SELECT COUNT(*) FROM %s", stateTable)).Scan(&count); err != nil {
		return
	}
	if count == 0 {
		r.conn.Exec(ctx, fmt.Sprintf("DROP TABLE IF EXISTS %s", stateTable)) //nolint:errcheck
	}
}

// resumeGate detects orphan shadow tables and prompts the operator.
func (r *MigrationRunner) resumeGate(ctx context.Context) error {
	rows, err := r.conn.Query(ctx, `
		SELECT tablename FROM pg_tables
		WHERE schemaname = 'public'
		AND tablename LIKE '_bakeoff_migration_%'
		AND tablename NOT LIKE '_bakeoff_migration_state'
	`)
	if err != nil {
		return err
	}
	defer rows.Close()

	var orphans []string
	for rows.Next() {
		var name string
		if err := rows.Scan(&name); err != nil {
			return err
		}
		orphans = append(orphans, name)
	}
	if len(orphans) == 0 {
		return nil
	}

	// Orphans found — surface for operator decision.
	log.Printf("[migrate] WARNING: found %d orphan shadow table(s) from a prior run:", len(orphans))
	for _, name := range orphans {
		log.Printf("  %s", name)
	}
	log.Printf("[migrate] Options: (c)ontinue, (r)ollback, (i)nspect")
	log.Printf("[migrate] Continuing automatically (use --rollback flag to drop orphans).")
	// Non-interactive mode: continue (resume semantics). CLI flag --rollback would drop here.
	// TODO(phase2b): wire --rollback flag to drop all orphan shadow tables.
	return nil
}

// loadSchemaVersion loads a schema_versions row by ID.
func (r *MigrationRunner) loadSchemaVersion(ctx context.Context, id int) (SchemaVersion, error) {
	var sv SchemaVersion
	err := r.conn.QueryRow(ctx, `
		SELECT schema_version_id, COALESCE(description,''), allow_migration,
		       schema_migration_script, record_migration_script
		FROM schema_versions WHERE schema_version_id = $1`, id).
		Scan(&sv.ID, &sv.Description, &sv.AllowMigration, &sv.SchemaMigrationScript, &sv.RecordMigrationScript)
	if err == pgx.ErrNoRows {
		return sv, fmt.Errorf("schema version %d not found", id)
	}
	return sv, err
}

// loadSchemaTables loads all schema_tables rows.
func (r *MigrationRunner) loadSchemaTables(ctx context.Context) ([]SchemaTable, error) {
	rows, err := r.conn.Query(ctx, `
		SELECT table_id::text, table_name, uuid_namespace::text,
		       current_version_id, deprecated_at
		FROM schema_tables ORDER BY created_at`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var tables []SchemaTable
	for rows.Next() {
		var t SchemaTable
		if err := rows.Scan(&t.TableID, &t.TableName, &t.UUIDNamespace, &t.CurrentVersionID, &t.DeprecatedAt); err != nil {
			return nil, err
		}
		tables = append(tables, t)
	}
	return tables, nil
}

// topologicalOrder returns schema_tables ordered by FK dependency (leaves first).
// Cycles (self-referential) are moved to end of list with a warning.
func (r *MigrationRunner) topologicalOrder(ctx context.Context, tables []SchemaTable) ([]SchemaTable, error) {
	// Build adjacency: table_name → set of FK-referenced table_names
	// using information_schema.referential_constraints + key_column_usage.
	deps := make(map[string]map[string]bool, len(tables))
	for _, t := range tables {
		deps[t.TableName] = make(map[string]bool)
	}

	fkRows, err := r.conn.Query(ctx, `
		SELECT
			tc.table_name   AS src,
			ccu.table_name  AS dst
		FROM information_schema.table_constraints tc
		JOIN information_schema.referential_constraints rc
			ON tc.constraint_name = rc.constraint_name
		JOIN information_schema.key_column_usage kcu
			ON tc.constraint_name = kcu.constraint_name
		JOIN information_schema.constraint_column_usage ccu
			ON rc.unique_constraint_name = ccu.constraint_name
		WHERE tc.constraint_type = 'FOREIGN KEY'
		AND tc.table_schema = 'public'
		GROUP BY tc.table_name, ccu.table_name`)
	if err != nil {
		return tables, nil // Degrade gracefully if query fails.
	}
	defer fkRows.Close()

	for fkRows.Next() {
		var src, dst string
		if err := fkRows.Scan(&src, &dst); err != nil {
			continue
		}
		if _, ok := deps[src]; ok {
			deps[src][dst] = true
		}
	}

	// Kahn's topological sort.
	inDegree := make(map[string]int, len(tables))
	for name, fdeps := range deps {
		if _, ok := inDegree[name]; !ok {
			inDegree[name] = 0
		}
		for dep := range fdeps {
			if dep != name { // Ignore self-refs for degree counting.
				inDegree[dep]++
			}
		}
	}

	// Collect zero-in-degree nodes (leaves, no dependents).
	var queue []string
	for _, t := range tables {
		if inDegree[t.TableName] == 0 {
			queue = append(queue, t.TableName)
		}
	}

	nameToTable := make(map[string]SchemaTable, len(tables))
	for _, t := range tables {
		nameToTable[t.TableName] = t
	}

	var ordered []SchemaTable
	visited := make(map[string]bool)
	for len(queue) > 0 {
		name := queue[0]
		queue = queue[1:]
		if visited[name] {
			continue
		}
		visited[name] = true
		if t, ok := nameToTable[name]; ok {
			ordered = append(ordered, t)
		}
		for dep := range deps[name] {
			if dep == name {
				continue
			}
			inDegree[dep]--
			if inDegree[dep] == 0 {
				queue = append(queue, dep)
			}
		}
	}

	// Any unvisited tables have cycles — append with warning.
	for _, t := range tables {
		if !visited[t.TableName] {
			log.Printf("[migrate] WARNING: table %q is in a FK cycle; migrated last (deferral queue not yet implemented)", t.TableName)
			ordered = append(ordered, t)
		}
	}

	return ordered, nil
}

// compileScript compiles a Tengo script with the given builtins pre-defined.
func compileScript(script string, builtins map[string]tengo.Object) (*tengo.Compiled, error) {
	s := tengo.NewScript([]byte(script))
	for name, obj := range builtins {
		if err := s.Add(name, obj); err != nil {
			return nil, fmt.Errorf("register builtin %q: %w", name, err)
		}
	}
	return s.Compile()
}

// sanitizeName replaces non-identifier characters with underscores.
func sanitizeName(name string) string {
	var b strings.Builder
	for _, c := range name {
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') {
			b.WriteRune(c)
		} else {
			b.WriteRune('_')
		}
	}
	return b.String()
}

// primaryKeyOf extracts the "id" field value from a row (best-effort).
func primaryKeyOf(row map[string]any) any {
	if v, ok := row["id"]; ok {
		return v
	}
	return "<unknown>"
}
