package migrate

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
)

// Namespaces match bench/constants.py so disk UUIDs and DB keys agree.
const (
	modelNamespace   = "dbc87c22-6b6d-4f89-9757-d186f0f62719"
	creatorNamespace = "efaf4f4e-cbd1-43f7-a535-f72079513d00"
)

// recordSchemaVersion matches SCHEMA_VERSION in bench/store.py.
const recordSchemaVersion = 1

type rowCounts map[string]int64

type ingestFunc func(ctx context.Context, tx pgx.Tx, path string, rec map[string]any) (rowCounts, error)

// Sync upserts JSON store records and results/*.json into schema/schema.sql
// tables. Each file is one transaction. A re-run inserts nothing.
func Sync(ctx context.Context, conn *pgx.Conn, dataDir, resultsDir string) error {
	steps := []struct {
		rel string
		fn  ingestFunc
	}{
		{"models", ingestModelFile},
		{"tasks", ingestTaskFile},
		{"prompts", ingestPromptFile},
		{"runners", ingestRunnerFile},
		{"runs", ingestRunFile},
		{filepath.Join("run_queue", "pending"), ingestQueueFile},
		{filepath.Join("run_queue", "completed"), ingestQueueFile},
	}
	for _, step := range steps {
		if err := ingestDir(ctx, conn, filepath.Join(dataDir, step.rel), step.fn); err != nil {
			return err
		}
	}
	if resultsDir == "" {
		return nil
	}
	return ingestDir(ctx, conn, resultsDir, ingestRunFile)
}

func ingestDir(ctx context.Context, conn *pgx.Conn, dir string, fn ingestFunc) error {
	matches, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return err
	}
	sort.Strings(matches)
	for _, path := range matches {
		if err := ingestFile(ctx, conn, path, fn); err != nil {
			return err
		}
	}
	return nil
}

func ingestFile(ctx context.Context, conn *pgx.Conn, path string, fn ingestFunc) error {
	rec, err := loadRecord(path)
	if err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	tx, err := conn.Begin(ctx)
	if err != nil {
		return fmt.Errorf("%s: begin: %w", path, err)
	}
	defer func() { _ = tx.Rollback(ctx) }()
	counts, err := fn(ctx, tx, path, rec)
	if err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("%s: commit: %w", path, err)
	}
	log.Printf("ingest %s: %s", path, formatCounts(counts))
	return nil
}

func loadRecord(path string) (map[string]any, error) {
	body, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return nil, err
	}
	var v any
	if err := json.Unmarshal(body, &v); err != nil {
		return nil, err
	}
	rec, ok := v.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("must contain a JSON object")
	}
	// Flat results files carry no schema_version; store records always do, and
	// a newer shape read with this version's field mapping would ingest wrong.
	if sv, ok := rec["schema_version"]; ok && sv != float64(recordSchemaVersion) {
		return nil, fmt.Errorf("unsupported schema_version %v (want %d)", sv, recordSchemaVersion)
	}
	return rec, nil
}

func formatCounts(c rowCounts) string {
	if len(c) == 0 {
		return "0 rows"
	}
	keys := make([]string, 0, len(c))
	var total int64
	for k, n := range c {
		keys = append(keys, k)
		total += n
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys)+1)
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", k, c[k]))
	}
	parts = append(parts, fmt.Sprintf("(%d rows)", total))
	return strings.Join(parts, " ")
}

func ingestModelFile(ctx context.Context, tx pgx.Tx, path string, rec map[string]any) (rowCounts, error) {
	counts := rowCounts{"creators": 0, "models": 0, "model_sources": 0}
	var creatorID any
	if raw, ok := rec["creator"].(map[string]any); ok {
		n, id, err := insertCreator(ctx, tx, raw)
		if err != nil {
			return counts, err
		}
		counts["creators"] = n
		creatorID = id
	}
	modelID, err := modelIDFrom(path, rec)
	if err != nil {
		return counts, err
	}
	n, err := insertModel(ctx, tx, modelID, creatorID, rec)
	if err != nil {
		return counts, err
	}
	counts["models"] = n
	n, err = insertSources(ctx, tx, modelID, rec)
	if err != nil {
		return counts, err
	}
	counts["model_sources"] = n
	return counts, nil
}

func insertCreator(ctx context.Context, tx pgx.Tx, rec map[string]any) (int64, uuid.UUID, error) {
	id := creatorIDFrom(rec)
	n, err := execCount(ctx, tx, `
		INSERT INTO creators (creator_id, name, display_name, homepage, service_identifiers, provisional)
		VALUES ($1, $2, $3, $4, $5, $6)
		ON CONFLICT (creator_id) DO NOTHING`,
		id,
		asString(rec, "name"),
		nullString(rec, "display_name"),
		nullString(rec, "homepage"),
		asJSONB(rec["service_identifiers"]),
		asBool(rec, "provisional"),
	)
	return n, id, err
}

func insertModel(ctx context.Context, tx pgx.Tx, id uuid.UUID, creatorID any, rec map[string]any) (int64, error) {
	name := asString(rec, "name")
	if name == "" {
		return 0, fmt.Errorf("name is required")
	}
	return execCount(ctx, tx, `
		INSERT INTO models (
			model_id, name, creator_id, model_hash,
			parameter_count_b, active_parameter_count_b,
			architecture_id, context_length_default, context_length_min, context_length_max,
			file_format_id, quantization_id,
			model_source_mtime, model_source_size,
			release_date, version, description,
			predecessor_model_id, provisional
		) VALUES (
			$1, $2, $3, $4,
			$5, $6,
			$7, $8, $9, $10,
			$11, $12,
			$13, $14,
			$15, $16, $17,
			$18, $19
		)
		ON CONFLICT (model_id) DO NOTHING`,
		id, name, creatorID, nullString(rec, "model_hash"),
		nullNumber(rec, "parameter_count_b"), nullNumber(rec, "active_parameter_count_b"),
		lookupNameID(ctx, tx, "SELECT architecture_id FROM model_architectures WHERE name = $1", asString(rec, "architecture")),
		nullInt(rec, "context_length_default"), nullInt(rec, "context_length_min"), nullInt(rec, "context_length_max"),
		lookupNameID(ctx, tx, "SELECT file_format_id FROM model_file_formats WHERE name = $1", asString(rec, "file_format")),
		lookupNameID(ctx, tx, "SELECT quantization_id FROM quantization_methods WHERE name = $1", strings.ToLower(asString(rec, "quantization"))),
		asTime(first(rec, "model_source_mtime")), nullInt64(rec, "model_source_size"),
		asDate(first(rec, "release_date")), nullString(rec, "version"), nullString(rec, "description"),
		asUUID(first(rec, "predecessor_model_id")), asBool(rec, "provisional"),
	)
}

func insertSources(ctx context.Context, tx pgx.Tx, modelID uuid.UUID, rec map[string]any) (int64, error) {
	raw, _ := rec["sources"].([]any)
	var total int64
	for _, item := range raw {
		src, ok := item.(map[string]any)
		if !ok {
			return total, fmt.Errorf("sources entries must be objects")
		}
		url := asString(src, "url")
		if url == "" {
			return total, fmt.Errorf("source url is required")
		}
		typeID := lookupNameID(ctx, tx, "SELECT source_type_id FROM source_types WHERE name = $1", asString(src, "source_type"))
		if typeID == nil {
			return total, fmt.Errorf("unknown source_type %q", asString(src, "source_type"))
		}
		n, err := execCount(ctx, tx, `
			INSERT INTO model_sources (model_id, source_type_id, url, source_metadata, updated)
			SELECT $1, $2, $3, $4, $5
			WHERE NOT EXISTS (
				SELECT 1 FROM model_sources s WHERE s.model_id = $1 AND s.url = $3
			)`,
			modelID, typeID, url, asJSONB(src["source_metadata"]), asTime(first(src, "updated")),
		)
		if err != nil {
			return total, err
		}
		total += n
	}
	return total, nil
}

func ingestTaskFile(ctx context.Context, tx pgx.Tx, path string, rec map[string]any) (rowCounts, error) {
	counts := rowCounts{"tasks": 0}
	hash := asString(rec, "natural_key_hash")
	if hash == "" {
		hash = strings.TrimSuffix(filepath.Base(path), ".json")
	}
	name := asString(rec, "name")
	if name == "" {
		name = hash
	}
	n, err := execCount(ctx, tx, `
		INSERT INTO tasks (name, category_id, parent_id, sort_order, description, grader_script, grader_script_commit, natural_key_hash)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		ON CONFLICT (natural_key_hash) DO NOTHING`,
		name,
		lookupNameID(ctx, tx, "SELECT category_id FROM task_categories WHERE name = $1", asString(rec, "category")),
		nullInt(rec, "parent_id"),
		asIntDefault(rec, "sort_order", 0),
		nullString(rec, "description"),
		nullString(rec, "grader_script"),
		nullString(rec, "grader_script_commit"),
		hash,
	)
	counts["tasks"] = n
	return counts, err
}

func ingestPromptFile(ctx context.Context, tx pgx.Tx, path string, rec map[string]any) (rowCounts, error) {
	counts := rowCounts{"prompts": 0}
	sha := asString(rec, "content_sha256")
	if sha == "" {
		sha = strings.TrimSuffix(filepath.Base(path), ".json")
	}
	taskID, err := resolveTaskID(ctx, tx, rec)
	if err != nil {
		return counts, err
	}
	filePath := asString(rec, "file_path")
	if filePath == "" {
		filePath = sha
	}
	gitHash := asString(rec, "git_commit_hash")
	if gitHash == "" {
		gitHash = "unknown"
	}
	n, err := execCount(ctx, tx, `
		INSERT INTO prompts (
			task_id, file_path, git_commit_hash, content_sha256, content_length_bytes,
			version, release_date, is_prerelease, difficulty, modified_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
		ON CONFLICT (content_sha256) DO NOTHING`,
		taskID, filePath, gitHash, sha, nullInt(rec, "content_length_bytes"),
		nullString(rec, "version"), asDate(first(rec, "release_date")), asBool(rec, "is_prerelease"),
		asIntDefault(rec, "difficulty", 0), asTime(first(rec, "modified_at")),
	)
	counts["prompts"] = n
	return counts, err
}

func ingestRunnerFile(ctx context.Context, tx pgx.Tx, path string, rec map[string]any) (rowCounts, error) {
	counts := rowCounts{"runners": 0}
	id := asString(rec, "runner_id")
	if id == "" {
		id = strings.TrimSuffix(filepath.Base(path), ".json")
	}
	status, err := runnerStatus(asString(rec, "status"))
	if err != nil {
		return counts, err
	}
	n, err := execCount(ctx, tx, `
		INSERT INTO runners (
			runner_id, public_key, hostname, process_id, effective_user,
			last_heartbeat, status, registered_at, started_at, description
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
		ON CONFLICT (runner_id) DO UPDATE SET
			hostname = EXCLUDED.hostname, process_id = EXCLUDED.process_id,
			effective_user = EXCLUDED.effective_user, last_heartbeat = EXCLUDED.last_heartbeat,
			status = EXCLUDED.status, started_at = EXCLUDED.started_at, description = EXCLUDED.description
		WHERE (runners.hostname, runners.process_id, runners.effective_user, runners.last_heartbeat,
			runners.status, runners.started_at, runners.description)
			IS DISTINCT FROM (EXCLUDED.hostname, EXCLUDED.process_id, EXCLUDED.effective_user,
			EXCLUDED.last_heartbeat, EXCLUDED.status, EXCLUDED.started_at, EXCLUDED.description)`,
		id,
		asString(rec, "public_key"),
		nullString(rec, "hostname"),
		nullInt(rec, "process_id"),
		nullString(rec, "effective_user"),
		asTime(first(rec, "last_heartbeat")),
		status,
		asTime(first(rec, "registered_at")),
		asTime(first(rec, "started_at")),
		nullString(rec, "description"),
	)
	counts["runners"] = n
	return counts, err
}

func ingestRunFile(ctx context.Context, tx pgx.Tx, path string, rec map[string]any) (rowCounts, error) {
	counts := rowCounts{"runs": 0}
	id := runIDFrom(path, rec)
	submitted := asTime(first(rec, "submitted_at", "created_at", "timestamp"))
	if submitted == nil {
		submitted = time.Now().UTC()
	}
	publisher := asString(rec, "publisher_id")
	if publisher == "" {
		publisher = "unknown"
	}
	var runnerID any
	if rid := asString(rec, "runner_id"); rid != "" {
		if runnerExists(ctx, tx, rid) {
			runnerID = rid
		}
	}
	n, err := execCount(ctx, tx, `
		INSERT INTO runs (run_id, submitted_at, publisher_id, runner_version, prompt_git_hash, runner_id, run_status)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		ON CONFLICT (run_id) DO UPDATE SET run_status = EXCLUDED.run_status
		WHERE runs.run_status IS DISTINCT FROM EXCLUDED.run_status`,
		id, submitted, publisher,
		nullString(rec, "runner_version"),
		nullString(rec, "prompt_git_hash"),
		runnerID,
		runStatus(asString(rec, "run_status")),
	)
	counts["runs"] = n
	if err != nil {
		return counts, err
	}
	m, err := ingestRunMetrics(ctx, tx, id, rec)
	counts["run_model_metrics"] = m
	return counts, err
}

type cellAgg struct {
	promptID int
	modelID  uuid.UUID
	sum      float64
	scored   int
	failure  string
	detail   string
}

// ingestRunMetrics rolls the harness's per-task records up to one row per
// (prompt, model), the grain of run_model_metrics. Records whose harness
// model or prompt id is not in the models/prompts tables are skipped and
// counted in the log.
func ingestRunMetrics(ctx context.Context, tx pgx.Tx, runID uuid.UUID, rec map[string]any) (int64, error) {
	cells, _ := rec["records"].([]any)
	if len(cells) == 0 {
		return 0, nil
	}
	harness, _ := rec["config"].(map[string]any)
	gitRef := asString(rec, "prompt_git_hash")
	if prov, ok := rec["provenance"].(map[string]any); ok && gitRef == "" {
		if g, ok := prov["git"].(map[string]any); ok {
			gitRef = asString(g, "sha")
		}
	}
	aggs := map[[2]string]*cellAgg{}
	var order [][2]string
	var skipped int
	for _, item := range cells {
		cell, ok := item.(map[string]any)
		if !ok {
			skipped++
			continue
		}
		key := [2]string{asString(cell, "prompt_id"), asString(cell, "model_id")}
		agg, seen := aggs[key]
		if !seen {
			modelID, okM := resolveModelKey(ctx, tx, key[1], harness)
			promptID, okP := resolvePromptKey(ctx, tx, key[0], harness, gitRef)
			if okM && okP {
				agg = &cellAgg{promptID: promptID, modelID: modelID}
				order = append(order, key)
			}
			aggs[key] = agg
		}
		if agg == nil {
			skipped++
			continue
		}
		if score, ok := cell["quality_heuristic"].(float64); ok {
			agg.sum += score
			agg.scored++
		}
		if code := asString(cell, "failure_code"); code != "" && agg.failure == "" {
			agg.failure, agg.detail = code, asString(cell, "failure_detail")
		}
	}
	var total int64
	for _, key := range order {
		agg := aggs[key]
		var score any
		if agg.scored > 0 {
			score = agg.sum / float64(agg.scored)
		}
		n, err := execCount(ctx, tx, `
			INSERT INTO run_model_metrics (run_id, prompt_id, model_id, score, failure_reason, failure_detail)
			VALUES ($1, $2, $3, $4, $5, $6)
			ON CONFLICT (run_id, prompt_id, model_id) DO UPDATE SET
				score = EXCLUDED.score, failure_reason = EXCLUDED.failure_reason,
				failure_detail = EXCLUDED.failure_detail
			WHERE (run_model_metrics.score, run_model_metrics.failure_reason, run_model_metrics.failure_detail)
				IS DISTINCT FROM (EXCLUDED.score, EXCLUDED.failure_reason, EXCLUDED.failure_detail)`,
			runID, agg.promptID, agg.modelID, score, nullString(map[string]any{"v": agg.failure}, "v"),
			nullString(map[string]any{"v": agg.detail}, "v"))
		if err != nil {
			return total, err
		}
		total += n
	}
	if skipped > 0 {
		log.Printf("run %s: skipped %d result cells with unresolvable model or prompt ids", runID, skipped)
	}
	return total, nil
}

// resolveModelKey maps a harness model id (config.yaml `id`, e.g.
// "llama3_1_8b_q4") to the models row named after its GGUF file, falling
// back to a direct name or UUID match.
func resolveModelKey(ctx context.Context, tx pgx.Tx, key string, cfg map[string]any) (uuid.UUID, bool) {
	name := key
	entries, _ := cfg["models"].([]any)
	for _, e := range entries {
		if m, ok := e.(map[string]any); ok && asString(m, "id") == key {
			file := asString(m, "filename")
			if file == "" {
				file = filepath.Base(asString(m, "gguf"))
			}
			name = strings.TrimSuffix(file, ".gguf")
			break
		}
	}
	// models.name is not unique (one GGUF filename can come from several
	// repos); attributing a score to an arbitrary twin would be silent misdata.
	rows, err := tx.Query(ctx, "SELECT model_id FROM models WHERE name = $1 OR model_id::text = $1 LIMIT 2", name)
	if err != nil {
		return uuid.Nil, false
	}
	ids, err := pgx.CollectRows(rows, pgx.RowTo[uuid.UUID])
	if err != nil || len(ids) != 1 {
		return uuid.Nil, false
	}
	return ids[0], true
}

// harnessTaskHash anchors prompt variants the harness defines per run: the
// prompts table requires a task, and a variant applies to every task.
const harnessTaskHash = "bakeoff:harness-prompt-variants"

// resolvePromptKey finds a stored prompt by content hash or path, else
// registers the run's prompt variant (config.prompts `id`, or the
// system-less "floor" cell) as a prompts row keyed by id plus system text.
func resolvePromptKey(ctx context.Context, tx pgx.Tx, key string, cfg map[string]any, gitRef string) (int, bool) {
	var id int
	if err := tx.QueryRow(ctx, "SELECT prompt_id FROM prompts WHERE content_sha256 = $1 OR file_path = $1 LIMIT 1", key).Scan(&id); err == nil {
		return id, true
	}
	system, found := "", key == "floor"
	entries, _ := cfg["prompts"].([]any)
	for _, e := range entries {
		if p, ok := e.(map[string]any); ok && asString(p, "id") == key {
			system, found = asString(p, "system"), true
			break
		}
	}
	if !found {
		return 0, false
	}
	if gitRef == "" {
		gitRef = "unknown"
	}
	sum := sha256.Sum256([]byte(key + "\x00" + system))
	sha := hex.EncodeToString(sum[:])
	err := tx.QueryRow(ctx, `
		WITH t AS (
			INSERT INTO tasks (name, natural_key_hash) VALUES ('harness prompt variants', $1)
			ON CONFLICT (natural_key_hash) DO UPDATE SET natural_key_hash = EXCLUDED.natural_key_hash
			RETURNING task_id)
		INSERT INTO prompts (task_id, file_path, git_commit_hash, content_sha256, content_length_bytes)
		SELECT task_id, $2, $3, $4, $5 FROM t
		ON CONFLICT (content_sha256) DO UPDATE SET content_sha256 = EXCLUDED.content_sha256
		RETURNING prompt_id`, harnessTaskHash, key, gitRef, sha, len(system)).Scan(&id)
	return id, err == nil
}

func ingestQueueFile(ctx context.Context, tx pgx.Tx, path string, rec map[string]any) (rowCounts, error) {
	counts := rowCounts{"run_queue": 0}
	// The runner's per-run lifecycle record shares this directory but names no
	// prompt; run_queue rows are per-prompt work items, so it has no row here.
	if first(rec, "prompt_id", "content_sha256") == nil {
		log.Printf("%s: skipped run lifecycle record without prompt_id", path)
		return counts, nil
	}
	qid, err := parseUUID(first(rec, "queue_id"), strings.TrimSuffix(filepath.Base(path), ".json"))
	if err != nil {
		return counts, fmt.Errorf("queue_id: %w", err)
	}
	runID, err := parseUUID(first(rec, "run_id"), "")
	if err != nil {
		return counts, fmt.Errorf("run_id: %w", err)
	}
	promptID, err := resolvePromptID(ctx, tx, rec)
	if err != nil {
		return counts, err
	}
	status, err := queueStatus(asString(rec, "status"))
	if err != nil {
		return counts, err
	}
	n, err := execCount(ctx, tx, `
		INSERT INTO run_queue (
			queue_id, run_id, prompt_id, priority, status, attempt_count, max_attempts,
			claimed_by, claimed_at, started_at, completed_at, error_detail, retry_after,
			source_file, created_at, updated_at
		) VALUES (
			$1, $2, $3, $4, $5, $6, $7,
			$8, $9, $10, $11, $12, $13,
			$14, COALESCE($15, NOW()), COALESCE($16, NOW())
		)
		ON CONFLICT (queue_id) DO UPDATE SET
			status = EXCLUDED.status, attempt_count = EXCLUDED.attempt_count,
			claimed_by = EXCLUDED.claimed_by, claimed_at = EXCLUDED.claimed_at,
			started_at = EXCLUDED.started_at, completed_at = EXCLUDED.completed_at,
			error_detail = EXCLUDED.error_detail, retry_after = EXCLUDED.retry_after,
			source_file = EXCLUDED.source_file, updated_at = EXCLUDED.updated_at
		WHERE (run_queue.status, run_queue.attempt_count, run_queue.claimed_by, run_queue.claimed_at,
			run_queue.started_at, run_queue.completed_at, run_queue.error_detail,
			run_queue.retry_after, run_queue.source_file)
			IS DISTINCT FROM (EXCLUDED.status, EXCLUDED.attempt_count, EXCLUDED.claimed_by,
			EXCLUDED.claimed_at, EXCLUDED.started_at, EXCLUDED.completed_at, EXCLUDED.error_detail,
			EXCLUDED.retry_after, EXCLUDED.source_file)`,
		qid, runID, promptID,
		asIntDefault(rec, "priority", 100),
		status,
		asIntDefault(rec, "attempt_count", 0),
		asIntDefault(rec, "max_attempts", 5),
		nullString(rec, "claimed_by"),
		asTime(first(rec, "claimed_at")),
		asTime(first(rec, "started_at")),
		asTime(first(rec, "completed_at")),
		nullString(rec, "error_detail"),
		asTime(first(rec, "retry_after")),
		nullString(rec, "source_file"),
		asTime(first(rec, "created_at")),
		asTime(first(rec, "updated_at")),
	)
	counts["run_queue"] = n
	return counts, err
}

func modelIDFrom(path string, rec map[string]any) (uuid.UUID, error) {
	stem := strings.TrimSuffix(filepath.Base(path), ".json")
	if id, err := uuid.Parse(stem); err == nil {
		return id, nil
	}
	if s := asString(rec, "uuid"); s != "" {
		return uuid.Parse(s)
	}
	ns := uuid.MustParse(modelNamespace)
	if h := asString(rec, "model_hash"); h != "" {
		return uuid.NewSHA1(ns, []byte(h)), nil
	}
	url := ""
	if sources, _ := rec["sources"].([]any); len(sources) > 0 {
		if src, ok := sources[0].(map[string]any); ok {
			url = asString(src, "url")
		}
	}
	params, err := canonicalNumber(rec["parameter_count_b"])
	if err != nil {
		return uuid.Nil, err
	}
	size, err := canonicalNumber(rec["model_source_size"])
	if err != nil {
		return uuid.Nil, err
	}
	key := url + "|" + params + "|" + size
	return uuid.NewSHA1(ns, []byte(key)), nil
}

func creatorIDFrom(rec map[string]any) uuid.UUID {
	ns := uuid.MustParse(creatorNamespace)
	if hp := asString(rec, "homepage"); hp != "" {
		return uuid.NewSHA1(ns, []byte(hp))
	}
	name := asString(rec, "display_name")
	if name == "" {
		name = asString(rec, "name")
	}
	return uuid.NewSHA1(ns, []byte(name))
}

func runIDFrom(path string, rec map[string]any) uuid.UUID {
	raw := asString(rec, "run_id")
	if raw == "" {
		raw = strings.TrimSuffix(filepath.Base(path), ".json")
	}
	if id, err := uuid.Parse(raw); err == nil {
		return id
	}
	// run_id defaults to config.yaml run.name, which every run shares; the
	// timestamp keeps separate runs apart while runs/ and results/ copies agree.
	if ts := asString(rec, "timestamp"); ts != "" {
		raw += "|" + ts
	}
	return uuid.NewSHA1(uuid.NameSpaceURL, []byte("bakeoff:run:"+raw))
}

func resolveTaskID(ctx context.Context, tx pgx.Tx, rec map[string]any) (int, error) {
	if v := nullInt(rec, "task_id"); v != nil {
		if id, ok := v.(int); ok {
			return id, nil
		}
	}
	hash := asString(rec, "natural_key_hash")
	if hash == "" {
		hash = asString(rec, "task_natural_key_hash")
	}
	if hash == "" {
		return 0, fmt.Errorf("prompt task_id is required")
	}
	var id int
	err := tx.QueryRow(ctx, "SELECT task_id FROM tasks WHERE natural_key_hash = $1", hash).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("task %q: %w", hash, err)
	}
	return id, nil
}

func resolvePromptID(ctx context.Context, tx pgx.Tx, rec map[string]any) (int, error) {
	if v := nullInt(rec, "prompt_id"); v != nil {
		if id, ok := v.(int); ok {
			return id, nil
		}
	}
	key := asString(rec, "prompt_id")
	if key == "" {
		key = asString(rec, "content_sha256")
	}
	if key == "" {
		return 0, fmt.Errorf("prompt_id is required")
	}
	var id int
	err := tx.QueryRow(ctx, `
		SELECT prompt_id FROM prompts
		WHERE content_sha256 = $1 OR file_path = $1`, key).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("prompt %q: %w", key, err)
	}
	return id, nil
}

func runnerExists(ctx context.Context, tx pgx.Tx, id string) bool {
	var n int
	err := tx.QueryRow(ctx, "SELECT 1 FROM runners WHERE runner_id = $1", id).Scan(&n)
	return err == nil
}

func lookupNameID(ctx context.Context, tx pgx.Tx, q, name string) any {
	if name == "" {
		return nil
	}
	var id int
	if err := tx.QueryRow(ctx, q, name).Scan(&id); err != nil {
		return nil
	}
	return id
}

func execCount(ctx context.Context, tx pgx.Tx, sql string, args ...any) (int64, error) {
	tag, err := tx.Exec(ctx, sql, args...)
	if err != nil {
		return 0, err
	}
	return tag.RowsAffected(), nil
}

func runnerStatus(s string) (string, error) {
	if s == "" {
		return "ACTIVE", nil
	}
	switch strings.ToUpper(s) {
	case "ACTIVE", "IDLE", "DEAD":
		return strings.ToUpper(s), nil
	default:
		return "", fmt.Errorf("invalid runner status %q", s)
	}
}

func queueStatus(s string) (string, error) {
	if s == "" {
		return "PENDING", nil
	}
	switch strings.ToUpper(s) {
	case "PENDING", "CLAIMED", "IN_PROGRESS", "COMPLETE", "FAILED", "CANCELLED":
		return strings.ToUpper(s), nil
	default:
		return "", fmt.Errorf("invalid queue status %q", s)
	}
}

func runStatus(s string) any {
	if s == "" {
		return nil
	}
	switch strings.ToLower(s) {
	case "complete", "incomplete", "failed":
		return strings.ToLower(s)
	default:
		return nil
	}
}

func canonicalNumber(v any) (string, error) {
	if v == nil {
		return "", nil
	}
	switch n := v.(type) {
	case bool:
		return "", fmt.Errorf("canonical_number: bool is not a number")
	case int:
		return strconv.Itoa(n), nil
	case int32:
		return strconv.FormatInt(int64(n), 10), nil
	case int64:
		return strconv.FormatInt(n, 10), nil
	case float64:
		if math.IsNaN(n) || math.IsInf(n, 0) {
			return "", fmt.Errorf("canonical_number: %v is not finite", n)
		}
		return strconv.FormatFloat(n, 'f', -1, 64), nil
	case json.Number:
		f, err := n.Float64()
		if err != nil {
			return "", err
		}
		return canonicalNumber(f)
	case string:
		if n == "" {
			return "", nil
		}
		f, err := strconv.ParseFloat(n, 64)
		if err != nil {
			return "", err
		}
		return canonicalNumber(f)
	default:
		return "", fmt.Errorf("canonical_number: unsupported type %T", v)
	}
}

func asString(m map[string]any, k string) string {
	v, ok := m[k]
	if !ok || v == nil {
		return ""
	}
	s, ok := v.(string)
	if ok {
		return s
	}
	return fmt.Sprint(v)
}

func nullString(m map[string]any, k string) any {
	s := asString(m, k)
	if s == "" {
		return nil
	}
	return s
}

func asBool(m map[string]any, k string) bool {
	v, ok := m[k]
	if !ok || v == nil {
		return false
	}
	b, ok := v.(bool)
	return ok && b
}

func nullNumber(m map[string]any, k string) any {
	v, ok := m[k]
	if !ok || v == nil {
		return nil
	}
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	case int64:
		return float64(n)
	case json.Number:
		f, err := n.Float64()
		if err != nil {
			return nil
		}
		return f
	default:
		return nil
	}
}

func nullInt(m map[string]any, k string) any {
	v, ok := m[k]
	if !ok || v == nil {
		return nil
	}
	return toInt(v)
}

func nullInt64(m map[string]any, k string) any {
	v := nullInt(m, k)
	if v == nil {
		return nil
	}
	if n, ok := v.(int); ok {
		return int64(n)
	}
	return v
}

func asIntDefault(m map[string]any, k string, def int) int {
	v := nullInt(m, k)
	if v == nil {
		return def
	}
	n, ok := v.(int)
	if !ok {
		return def
	}
	return n
}

func toInt(v any) any {
	switch n := v.(type) {
	case float64:
		if n != math.Trunc(n) {
			return nil
		}
		return int(n)
	case int:
		return n
	case int32:
		return int(n)
	case int64:
		return int(n)
	case json.Number:
		i, err := n.Int64()
		if err != nil {
			return nil
		}
		return int(i)
	case string:
		i, err := strconv.Atoi(n)
		if err != nil {
			return nil
		}
		return i
	default:
		return nil
	}
}

func first(m map[string]any, keys ...string) any {
	for _, k := range keys {
		if v, ok := m[k]; ok && v != nil && v != "" {
			return v
		}
	}
	return nil
}

func asTime(v any) any {
	s := stringify(v)
	if s == "" {
		return nil
	}
	layouts := []string{
		time.RFC3339Nano,
		time.RFC3339,
		"2006-01-02T15:04:05Z",
		"2006-01-02T15:04:05",
		"20060102-150405",
		"2006-01-02",
	}
	for _, layout := range layouts {
		if t, err := time.Parse(layout, s); err == nil {
			return t.UTC()
		}
	}
	if t, err := time.Parse("2006-01-02T15:04:05.999999Z07:00", s); err == nil {
		return t.UTC()
	}
	return nil
}

func asDate(v any) any {
	t := asTime(v)
	if t == nil {
		return nil
	}
	tm, ok := t.(time.Time)
	if !ok {
		return nil
	}
	return tm.Format("2006-01-02")
}

func asUUID(v any) any {
	s := stringify(v)
	if s == "" {
		return nil
	}
	id, err := uuid.Parse(s)
	if err != nil {
		return nil
	}
	return id
}

func parseUUID(v any, fallback string) (uuid.UUID, error) {
	s := stringify(v)
	if s == "" {
		s = fallback
	}
	if s == "" {
		return uuid.Nil, fmt.Errorf("missing uuid")
	}
	return uuid.Parse(s)
}

func stringify(v any) string {
	if v == nil {
		return ""
	}
	s, ok := v.(string)
	if ok {
		return s
	}
	return fmt.Sprint(v)
}

func asJSONB(v any) any {
	if v == nil {
		return nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	return b
}
