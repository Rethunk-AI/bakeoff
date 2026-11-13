package migrate

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
)

func TestRunHardwareMetricsSQL_UpsertsOnRunID(t *testing.T) {
	sql := strings.Join(strings.Fields(runHardwareMetricsSQL()), " ")
	if !strings.Contains(sql, "ON CONFLICT (run_id) DO UPDATE") {
		t.Fatalf("hardware metrics SQL must upsert on run_id, got %q", sql)
	}
	if strings.Contains(sql, "DO NOTHING") {
		t.Fatalf("hardware metrics SQL skips on conflict instead of updating: %q", sql)
	}
}

func TestSync_IngestIsIdempotent(t *testing.T) {
	conn, ctx := testDB(t)
	dir := writeFixtureStore(t)
	results := filepath.Join(dir, "results")

	var buf bytes.Buffer
	log.SetOutput(&buf)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })

	if err := Sync(ctx, conn, dir, results); err != nil {
		t.Fatalf("sync: %v", err)
	}
	if got := countTable(t, ctx, conn, "models"); got != 1 {
		t.Fatalf("models after ingest: %d", got)
	}
	if got := countTable(t, ctx, conn, "creators"); got != 1 {
		t.Fatalf("creators after ingest: %d", got)
	}
	if got := countTable(t, ctx, conn, "model_sources"); got != 1 {
		t.Fatalf("model_sources after ingest: %d", got)
	}
	if got := countTable(t, ctx, conn, "runs"); got != 1 {
		t.Fatalf("runs after ingest: %d", got)
	}
	firstLog := buf.String()
	if !strings.Contains(firstLog, "models=1") || !strings.Contains(firstLog, "creators=1") {
		t.Fatalf("first ingest log = %q", firstLog)
	}

	buf.Reset()
	if err := Sync(ctx, conn, dir, results); err != nil {
		t.Fatalf("re-sync: %v", err)
	}
	if got := countTable(t, ctx, conn, "models"); got != 1 {
		t.Fatalf("models after re-ingest: %d", got)
	}
	if got := countTable(t, ctx, conn, "creators"); got != 1 {
		t.Fatalf("creators after re-ingest: %d", got)
	}
	if got := countTable(t, ctx, conn, "model_sources"); got != 1 {
		t.Fatalf("model_sources after re-ingest: %d", got)
	}
	if got := countTable(t, ctx, conn, "runs"); got != 1 {
		t.Fatalf("runs after re-ingest: %d", got)
	}
	secondLog := buf.String()
	if !strings.Contains(secondLog, "(0 rows)") {
		t.Fatalf("re-ingest log missing 0 rows: %q", secondLog)
	}
}

func TestSync_MalformedFileReportsPathAndWritesNothing(t *testing.T) {
	conn, ctx := testDB(t)
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "models")
	if err := os.MkdirAll(modelDir, 0o750); err != nil {
		t.Fatal(err)
	}
	bad := filepath.Join(modelDir, "broken.json")
	if err := os.WriteFile(bad, []byte("{"), 0o600); err != nil {
		t.Fatal(err)
	}

	err := Sync(t.Context(), conn, dir, filepath.Join(dir, "results"))
	if err == nil {
		t.Fatal("expected error for malformed file")
	}
	if !strings.Contains(err.Error(), bad) {
		t.Fatalf("error %q does not name %s", err, bad)
	}
	if got := countTable(t, ctx, conn, "models"); got != 0 {
		t.Fatalf("malformed file wrote models=%d", got)
	}
	if got := countTable(t, ctx, conn, "creators"); got != 0 {
		t.Fatalf("malformed file wrote creators=%d", got)
	}
	if got := countTable(t, ctx, conn, "model_sources"); got != 0 {
		t.Fatalf("malformed file wrote model_sources=%d", got)
	}
}

func testDB(t *testing.T) (*pgx.Conn, context.Context) {
	t.Helper()
	dsn := os.Getenv("DATABASE_URL")
	if dsn == "" {
		t.Skip("DATABASE_URL not set")
	}
	ctx := t.Context()
	admin, err := pgx.Connect(ctx, dsn)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	t.Cleanup(func() { _ = admin.Close(context.WithoutCancel(ctx)) })

	name := fmt.Sprintf("bakeoff_sync_%d", time.Now().UnixNano())
	ident := pgx.Identifier{name}.Sanitize()
	if _, err := admin.Exec(ctx, "CREATE DATABASE "+ident); err != nil {
		t.Fatalf("create database: %v", err)
	}
	t.Cleanup(func() {
		_, _ = admin.Exec(context.WithoutCancel(ctx), "DROP DATABASE IF EXISTS "+ident+" WITH (FORCE)")
	})

	cfg, err := pgx.ParseConfig(dsn)
	if err != nil {
		t.Fatalf("parse dsn: %v", err)
	}
	cfg.Database = name
	conn, err := pgx.ConnectConfig(ctx, cfg)
	if err != nil {
		t.Fatalf("connect test db: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close(context.WithoutCancel(ctx)) })
	applySchema(t, ctx, conn)
	return conn, ctx
}

func applySchema(t *testing.T, ctx context.Context, conn *pgx.Conn) {
	t.Helper()
	ddl, err := os.ReadFile("../schema/schema.sql")
	if err != nil {
		t.Fatalf("read schema: %v", err)
	}
	results, err := conn.PgConn().Exec(ctx, string(ddl)).ReadAll()
	if err != nil {
		t.Fatalf("apply schema: %v", err)
	}
	for _, result := range results {
		if result.Err != nil {
			t.Fatalf("apply schema: %v", result.Err)
		}
	}
}

func writeFixtureStore(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	modelNS := uuid.MustParse(modelNamespace)
	modelID := uuid.NewSHA1(modelNS, []byte("sha256:deadbeef1234"))
	modelDir := filepath.Join(dir, "models")
	if err := os.MkdirAll(modelDir, 0o750); err != nil {
		t.Fatal(err)
	}
	model := map[string]any{
		"schema_version":           1,
		"name":                     "TestModel-Q4_K_M",
		"model_hash":               "sha256:deadbeef1234",
		"parameter_count_b":        7.0,
		"active_parameter_count_b": 7.0,
		"architecture":             "Dense",
		"file_format":              "GGUF",
		"quantization":             "q4_k_m",
		"context_length_default":   4096,
		"provisional":              false,
		"creator": map[string]any{
			"name":         "test_org",
			"display_name": "Test Org",
			"homepage":     "https://example.com",
			"provisional":  false,
		},
		"sources": []any{
			map[string]any{
				"source_type":     "huggingface",
				"url":             "https://huggingface.co/test_org/TestModel-GGUF",
				"source_metadata": map[string]any{"hf_repo": "test_org/TestModel-GGUF"},
			},
		},
	}
	writeJSON(t, filepath.Join(modelDir, modelID.String()+".json"), model)

	runID := uuid.MustParse("11111111-1111-1111-1111-111111111111")
	results := filepath.Join(dir, "results")
	if err := os.MkdirAll(results, 0o750); err != nil {
		t.Fatal(err)
	}
	writeJSON(t, filepath.Join(results, "run-fixture.json"), map[string]any{
		"run_id":       runID.String(),
		"submitted_at": "2026-01-01T00:00:00Z",
		"publisher_id": "test",
		"run_status":   "complete",
	})
	return dir
}

func writeJSON(t *testing.T, path string, v any) {
	t.Helper()
	body, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(body, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
}

func countTable(t *testing.T, ctx context.Context, conn *pgx.Conn, table string) int {
	t.Helper()
	var n int
	q := "SELECT COUNT(*) FROM " + pgx.Identifier{table}.Sanitize()
	if err := conn.QueryRow(ctx, q).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

func TestSync_RefreshesMutableState(t *testing.T) {
	conn, ctx := testDB(t)
	dir := writeFixtureStore(t)
	results := filepath.Join(dir, "results")
	runFile := filepath.Join(results, "run-fixture.json")
	writeJSON(t, runFile, map[string]any{
		"run_id":       "11111111-1111-1111-1111-111111111111",
		"submitted_at": "2026-01-01T00:00:00Z",
		"publisher_id": "test",
		"run_status":   "incomplete",
	})
	if err := Sync(ctx, conn, dir, results); err != nil {
		t.Fatalf("sync: %v", err)
	}
	writeJSON(t, runFile, map[string]any{
		"run_id":       "11111111-1111-1111-1111-111111111111",
		"submitted_at": "2026-01-01T00:00:00Z",
		"publisher_id": "test",
		"run_status":   "complete",
	})
	if err := Sync(ctx, conn, dir, results); err != nil {
		t.Fatalf("re-sync: %v", err)
	}
	var status string
	if err := conn.QueryRow(ctx, "SELECT run_status::text FROM runs").Scan(&status); err != nil {
		t.Fatal(err)
	}
	if status != "complete" {
		t.Fatalf("run_status after resume = %q, want complete", status)
	}
}

func TestSync_IngestsPerCellResults(t *testing.T) {
	conn, ctx := testDB(t)
	dir := writeFixtureStore(t)
	for _, sub := range []string{"tasks", "prompts"} {
		if err := os.MkdirAll(filepath.Join(dir, sub), 0o750); err != nil {
			t.Fatal(err)
		}
	}
	writeJSON(t, filepath.Join(dir, "tasks", "t1.json"), map[string]any{"name": "t1", "natural_key_hash": "t1"})
	writeJSON(t, filepath.Join(dir, "prompts", "plain.json"), map[string]any{
		"content_sha256": "plain", "task_natural_key_hash": "t1", "file_path": "plain",
	})
	results := filepath.Join(dir, "results")
	writeJSON(t, filepath.Join(results, "run-fixture.json"), map[string]any{
		"run_id": "11111111-1111-1111-1111-111111111111", "submitted_at": "2026-01-01T00:00:00Z",
		"publisher_id": "test", "run_status": "complete",
		"records": []any{
			map[string]any{"model_id": "TestModel-Q4_K_M", "prompt_id": "plain", "quality_heuristic": 1.0},
			map[string]any{"model_id": "TestModel-Q4_K_M", "prompt_id": "plain", "quality_heuristic": 0.5},
			map[string]any{"model_id": "ghost", "prompt_id": "plain", "quality_heuristic": 1.0},
		},
	})
	var buf bytes.Buffer
	log.SetOutput(&buf)
	t.Cleanup(func() { log.SetOutput(os.Stderr) })
	if err := Sync(ctx, conn, dir, results); err != nil {
		t.Fatalf("sync: %v", err)
	}
	var n int
	var score float64
	if err := conn.QueryRow(ctx, "SELECT COUNT(*), COALESCE(MAX(score),0) FROM run_model_metrics").Scan(&n, &score); err != nil {
		t.Fatal(err)
	}
	if n != 1 || score != 0.75 {
		t.Fatalf("run_model_metrics rows=%d score=%v, want 1 / 0.75", n, score)
	}
	if !strings.Contains(buf.String(), "skipped 1 result cells") {
		t.Fatalf("missing skip log: %q", buf.String())
	}
}

func TestSync_IngestsRealShapedRunWithoutPromptStore(t *testing.T) {
	conn, ctx := testDB(t)
	dir := writeFixtureStore(t)
	results := filepath.Join(dir, "results")
	cell := func(model, prompt string, q float64) map[string]any {
		return map[string]any{"model_id": model, "prompt_id": prompt, "task_id": "t1", "quality_heuristic": q}
	}
	writeJSON(t, filepath.Join(results, "run-real.json"), map[string]any{
		"run_id": "11111111-1111-1111-1111-111111111111", "submitted_at": "2026-01-01T00:00:00Z",
		"publisher_id": "test", "run_status": "complete",
		"config": map[string]any{
			"models":  []any{map[string]any{"id": "test_model_q4", "filename": "TestModel-Q4_K_M.gguf"}},
			"prompts": []any{map[string]any{"id": "plain", "system": "Answer concisely."}},
		},
		"records": []any{
			cell("test_model_q4", "plain", 1.0), cell("test_model_q4", "plain", 0.5),
			cell("test_model_q4", "floor", 1.0), cell("ghost", "plain", 1.0),
		},
	})
	err := Sync(ctx, conn, dir, results)
	if err == nil {
		t.Fatal("expected error when prompt row is missing")
	}
	if !strings.Contains(err.Error(), `prompt "plain" not found`) {
		t.Fatalf("error %q, want prompt \"plain\" not found", err)
	}
}

func TestSync_SkipsRunnerLifecycleQueueRecords(t *testing.T) {
	conn, ctx := testDB(t)
	dir := writeFixtureStore(t)
	completed := filepath.Join(dir, "run_queue", "completed")
	if err := os.MkdirAll(completed, 0o750); err != nil {
		t.Fatal(err)
	}
	qid := uuid.New()
	writeJSON(t, filepath.Join(completed, qid.String()+".json"), map[string]any{
		"queue_id": qid.String(),
		"run_id":   "ab-run-001",
		"status":   "error",
	})
	if err := Sync(ctx, conn, dir, filepath.Join(dir, "results")); err != nil {
		t.Fatalf("sync: %v", err)
	}
	if got := countTable(t, ctx, conn, "run_queue"); got != 0 {
		t.Fatalf("run_queue rows = %d, want 0", got)
	}
}

func TestSync_KeepsRunsThatShareAConfigName(t *testing.T) {
	conn, ctx := testDB(t)
	dir := writeFixtureStore(t)
	results := filepath.Join(dir, "results")
	for _, ts := range []string{"20261101-000000", "20261102-000000"} {
		writeJSON(t, filepath.Join(results, "run-"+ts+".json"), map[string]any{
			"run_id":     "ab-run-001",
			"timestamp":  ts,
			"run_status": "complete",
		})
	}
	runs := filepath.Join(dir, "runs")
	if err := os.MkdirAll(runs, 0o750); err != nil {
		t.Fatal(err)
	}
	writeJSON(t, filepath.Join(runs, "ab-run-001.json"), map[string]any{
		"run_id":     "ab-run-001",
		"timestamp":  "20261102-000000",
		"run_status": "complete",
	})
	if err := Sync(ctx, conn, dir, results); err != nil {
		t.Fatalf("sync: %v", err)
	}
	// The fixture's own UUID run plus one row per distinct timestamp.
	if got := countTable(t, ctx, conn, "runs"); got != 3 {
		t.Fatalf("runs = %d, want 3", got)
	}
}

func TestSync_SkipsCellsWhoseModelNameIsAmbiguous(t *testing.T) {
	conn, ctx := testDB(t)
	dir := writeFixtureStore(t)
	for _, sub := range []string{"tasks", "prompts"} {
		if err := os.MkdirAll(filepath.Join(dir, sub), 0o750); err != nil {
			t.Fatal(err)
		}
	}
	writeJSON(t, filepath.Join(dir, "tasks", "t1.json"), map[string]any{"name": "t1", "natural_key_hash": "t1"})
	writeJSON(t, filepath.Join(dir, "prompts", "floor.json"), map[string]any{
		"content_sha256": "floor", "task_natural_key_hash": "t1", "file_path": "floor",
	})
	twin := uuid.NewSHA1(uuid.MustParse(modelNamespace), []byte("sha256:otherrepo"))
	writeJSON(t, filepath.Join(dir, "models", twin.String()+".json"), map[string]any{
		"name": "TestModel-Q4_K_M", "model_hash": "sha256:otherrepo",
	})
	results := filepath.Join(dir, "results")
	writeJSON(t, filepath.Join(results, "run-fixture.json"), map[string]any{
		"run_id": "11111111-1111-1111-1111-111111111111", "submitted_at": "2026-01-01T00:00:00Z",
		"publisher_id": "test", "run_status": "complete",
		"records": []any{
			map[string]any{"model_id": "TestModel-Q4_K_M", "prompt_id": "floor", "quality_heuristic": 1.0},
		},
	})
	if err := Sync(ctx, conn, dir, results); err != nil {
		t.Fatalf("sync: %v", err)
	}
	if got := countTable(t, ctx, conn, "run_model_metrics"); got != 0 {
		t.Fatalf("run_model_metrics = %d, want 0: two models share the name", got)
	}
}

func TestSync_RejectsUnknownRecordSchemaVersion(t *testing.T) {
	conn, ctx := testDB(t)
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "models")
	if err := os.MkdirAll(modelDir, 0o750); err != nil {
		t.Fatal(err)
	}
	writeJSON(t, filepath.Join(modelDir, uuid.NewString()+".json"), map[string]any{
		"schema_version": 2, "name": "FutureShape",
	})
	err := Sync(ctx, conn, dir, "")
	if err == nil || !strings.Contains(err.Error(), "schema_version") {
		t.Fatalf("sync err = %v, want schema_version error", err)
	}
	if got := countTable(t, ctx, conn, "models"); got != 0 {
		t.Fatalf("models = %d, want 0", got)
	}
}
