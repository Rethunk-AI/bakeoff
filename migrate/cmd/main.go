// bakeoff-migrate — migration runner CLI.
// Usage: bakeoff-migrate [flags] <schema_version_id>
//
// Design: bakeoff#27 (Phase 1 sign-off 050558ZJUN26).
// Phase 1 gate: squirrel + pgx/v5 + Tengo scripting engine.
// Phase 2b: batch sizing, shadow table, resume — implemented in runner.go.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/jackc/pgx/v5"

	migrate "github.com/Rethunk-AI/bakeoff/migrate"
)

func main() {
	fs := flag.NewFlagSet("bakeoff-migrate", flag.ExitOnError)

	var (
		batchSize   = fs.Int("batch-size", 0, "batch size: N>0 fixed, 0=dynamic (default), -1=invalid")
		maxRejects  = fs.Int("max-rejects", 10, "abort after N rejects (0=first, -1=collect all)")
		ignoreFatal = fs.Bool("ignore-is-fatal", false, "treat any ignoreRecord as a fatal error")
		dryRun      = fs.Bool("dry-run", false, "execute scripts without committing; report counts only")
		procTarget  = fs.Float64("batch-target-processing-time", 20.0, "target seconds per batch processing phase")
		commitTarget = fs.Float64("batch-target-commit-time", 1.0, "target seconds per batch commit phase")
		dsn         = fs.String("dsn", "", "PostgreSQL DSN (default: DATABASE_URL env var)")
	)

	if err := fs.Parse(os.Args[1:]); err != nil {
		log.Fatalf("parse flags: %v", err)
	}

	if *batchSize == -1 {
		log.Fatalf("--batch-size -1 is invalid; use 0 for dynamic sizing or N>0 for a fixed batch size")
	}

	if fs.NArg() != 1 {
		fmt.Fprintf(os.Stderr, "Usage: bakeoff-migrate [flags] <schema_version_id>\n\n")
		fs.PrintDefaults()
		os.Exit(1)
	}

	schemaVersionID, err := strconv.Atoi(fs.Arg(0))
	if err != nil || schemaVersionID <= 0 {
		log.Fatalf("schema_version_id must be a positive integer, got %q", fs.Arg(0))
	}

	// Resolve DSN.
	connStr := *dsn
	if connStr == "" {
		connStr = os.Getenv("DATABASE_URL")
	}
	if connStr == "" {
		log.Fatalf("DATABASE_URL not set and --dsn not provided")
	}

	ctx := context.Background()

	conn, err := pgx.Connect(ctx, connStr)
	if err != nil {
		log.Fatalf("connect to database: %v", err)
	}
	defer conn.Close(ctx)

	cfg := migrate.Config{
		BatchSize:            *batchSize,
		MaxRejects:           *maxRejects,
		IgnoreIsFatal:        *ignoreFatal,
		DryRun:               *dryRun,
		ProcessingTimeTarget: *procTarget,
		CommitTimeTarget:     *commitTarget,
	}

	runner := migrate.NewRunner(conn, cfg)

	log.Printf("[bakeoff-migrate] starting migration for schema version %d (dry-run=%v)", schemaVersionID, *dryRun)

	if err := runner.Run(ctx, schemaVersionID); err != nil {
		log.Fatalf("[bakeoff-migrate] FAILED: %v", err)
	}

	log.Printf("[bakeoff-migrate] DONE")
}
