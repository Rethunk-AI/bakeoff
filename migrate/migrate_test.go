package migrate

// migrate_test.go — unit tests for the migrate package.
// Covers: Record, type-conversion helpers, script compilation, built-in
// functions (hash/uuid_5/now/field/getField/setField/deleteField/rejectRecord/
// ignoreRecord), column-def builders, errMap, sanitizeName, primaryKeyOf,
// DefaultConfig, parseSelectArgs, applyWhere.
// DB-dependent paths (runner.Run, fetchBatch, shadow table ops) require a live
// PostgreSQL connection and are not covered here.

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/Masterminds/squirrel"
	"github.com/d5/tengo/v2"
)

// ─── helpers ──────────────────────────────────────────────────────────────────

// squirrelSelect returns a minimal squirrel SelectBuilder for applyWhere tests.
func squirrelSelect() squirrel.SelectBuilder {
	return squirrel.StatementBuilder.PlaceholderFormat(squirrel.Dollar).
		Select("*").From("t")
}

// noDBBuiltins returns record built-ins with nil conn (safe for tests that
// don't call select/selectFirst/selectOne/rawSQL).
func noDBBuiltins(rec *Record) map[string]tengo.Object {
	return recordBuiltins(nil, nil, rec)
}

// runTengo compiles and runs a script; fatal on error.
func runTengo(t *testing.T, script string, builtins map[string]tengo.Object) *tengo.Compiled {
	t.Helper()
	compiled, err := compileScript(script, builtins)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	if err := compiled.Run(); err != nil {
		t.Fatalf("run: %v", err)
	}
	return compiled
}

// getStr retrieves a *tengo.String value from a compiled variable.
func getStr(t *testing.T, compiled *tengo.Compiled, name string) string {
	t.Helper()
	v := compiled.Get(name)
	s, ok := v.Object().(*tengo.String)
	if !ok {
		t.Fatalf("%s: expected *tengo.String, got %T", name, v.Object())
	}
	return s.Value
}

// ─── Record ───────────────────────────────────────────────────────────────────

func TestRecord_GetSetDelete(t *testing.T) {
	rec := NewRecord(map[string]any{"a": "original", "b": int64(42)})

	if got := rec.Get("a"); got != "original" {
		t.Fatalf("Get(a) = %v, want original", got)
	}
	rec.Set("a", "overwritten")
	if got := rec.Get("a"); got != "overwritten" {
		t.Fatalf("Get(a) after Set = %v, want overwritten", got)
	}
	rec.Delete("a")
	merged := rec.MergedRow()
	if _, ok := merged["a"]; ok {
		t.Fatal("deleted field 'a' should be absent from MergedRow")
	}
	if merged["b"] != int64(42) {
		t.Fatalf("untouched field b = %v, want 42", merged["b"])
	}
}

func TestRecord_SetThenDelete(t *testing.T) {
	// Set then Delete: field absent.
	rec := NewRecord(map[string]any{"x": "keep"})
	rec.Set("x", "new")
	rec.Delete("x")
	if _, ok := rec.MergedRow()["x"]; ok {
		t.Fatal("field set then deleted should be absent from MergedRow")
	}
}

func TestRecord_DeleteThenSet(t *testing.T) {
	// Delete then Set: delete-set cleared, field present with new value.
	rec := NewRecord(map[string]any{"x": "old"})
	rec.Delete("x")
	rec.Set("x", "restored")
	merged := rec.MergedRow()
	if merged["x"] != "restored" {
		t.Fatalf("field deleted then set: want restored, got %v", merged["x"])
	}
}

func TestRecord_RejectIgnore(t *testing.T) {
	rec := NewRecord(nil)
	if rec.rejected || rec.ignored {
		t.Fatal("new record must not be rejected or ignored")
	}
	rec.Reject("bad data")
	if !rec.rejected || rec.rejectMsg != "bad data" {
		t.Fatalf("Reject: rejected=%v msg=%q", rec.rejected, rec.rejectMsg)
	}

	rec2 := NewRecord(nil)
	rec2.Ignore("skip it")
	if !rec2.ignored || rec2.ignoreMsg != "skip it" {
		t.Fatalf("Ignore: ignored=%v msg=%q", rec2.ignored, rec2.ignoreMsg)
	}
}

func TestRecord_MergedRowDirtyOverridesOriginal(t *testing.T) {
	rec := NewRecord(map[string]any{"k": "v1", "stable": true})
	rec.Set("k", "v2")
	merged := rec.MergedRow()
	if merged["k"] != "v2" {
		t.Fatalf("dirty field should override original; got %v", merged["k"])
	}
	if merged["stable"] != true {
		t.Fatal("untouched field must remain in MergedRow")
	}
}

func TestRecord_ToTengoMap(t *testing.T) {
	rec := NewRecord(map[string]any{"a": "v1", "b": int64(2)})
	rec.Set("a", "v2")
	rec.Delete("b")
	m := rec.toTengoMap()
	s, ok := m.Value["a"].(*tengo.String)
	if !ok || s.Value != "v2" {
		t.Errorf("toTengoMap: a should be v2, got %v", m.Value["a"])
	}
	if _, ok := m.Value["b"]; ok {
		t.Error("toTengoMap: deleted field b should be absent")
	}
}

// ─── goToTengo / tengoToGo ────────────────────────────────────────────────────

func TestGoToTengoTengoToGo_RoundTrip(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want any
	}{
		{"nil", nil, nil},
		{"bool true", true, true},
		{"bool false", false, false},
		{"int", int(7), int64(7)},
		{"int32", int32(32), int64(32)},
		{"int64", int64(64), int64(64)},
		{"float64", float64(3.14), float64(3.14)},
		{"string", "hello", "hello"},
		{"bytes", []byte("world"), []byte("world")},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := tengoToGo(goToTengo(tc.in))
			switch exp := tc.want.(type) {
			case nil:
				if got != nil {
					t.Fatalf("want nil, got %v", got)
				}
			case float64:
				gf, ok := got.(float64)
				if !ok || math.Abs(gf-exp) > 1e-9 {
					t.Fatalf("want %v, got %v", exp, got)
				}
			default:
				if fmt.Sprintf("%v", got) != fmt.Sprintf("%v", exp) {
					t.Fatalf("want %v, got %v", exp, got)
				}
			}
		})
	}
}

func TestGoToTengo_Float32(t *testing.T) {
	// float32 round-trips through float64.
	obj := goToTengo(float32(1.5))
	f, ok := obj.(*tengo.Float)
	if !ok {
		t.Fatalf("float32 should produce *tengo.Float, got %T", obj)
	}
	if math.Abs(f.Value-1.5) > 1e-4 {
		t.Errorf("float32 value: want ~1.5, got %v", f.Value)
	}
}

func TestGoToTengo_NestedMapAndArray(t *testing.T) {
	in := map[string]any{
		"arr": []any{"a", int64(1)},
		"sub": map[string]any{"deep": "val"},
	}
	out := tengoToGo(goToTengo(in))
	m, ok := out.(map[string]any)
	if !ok {
		t.Fatalf("expected map[string]any, got %T", out)
	}
	arr, ok := m["arr"].([]any)
	if !ok || len(arr) != 2 {
		t.Fatalf("nested array round-trip: %v", m["arr"])
	}
	sub, ok := m["sub"].(map[string]any)
	if !ok || sub["deep"] != "val" {
		t.Fatalf("nested map round-trip: %v", m["sub"])
	}
}

func TestGoToTengo_UnknownFallback(t *testing.T) {
	type custom struct{ N int }
	obj := goToTengo(custom{N: 99})
	s, ok := obj.(*tengo.String)
	if !ok {
		t.Fatalf("unknown type should produce *tengo.String, got %T", obj)
	}
	if !strings.Contains(s.Value, "99") {
		t.Errorf("stringified unknown should contain '99', got %q", s.Value)
	}
}

func TestTengoToGo_UndefinedIsNil(t *testing.T) {
	if got := tengoToGo(tengo.UndefinedValue); got != nil {
		t.Fatalf("UndefinedValue → nil, got %v", got)
	}
}

// ─── sanitizeName ─────────────────────────────────────────────────────────────

func TestSanitizeName(t *testing.T) {
	cases := []struct{ in, want string }{
		{"users", "users"},
		{"run_model_metrics", "run_model_metrics"},
		{"my-table.name", "my_table_name"},
		{"schema version 1", "schema_version_1"},
		{"ALL_CAPS", "ALL_CAPS"},
	}
	for _, tc := range cases {
		if got := sanitizeName(tc.in); got != tc.want {
			t.Errorf("sanitizeName(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// ─── primaryKeyOf ─────────────────────────────────────────────────────────────

func TestPrimaryKeyOf_Present(t *testing.T) {
	row := map[string]any{"id": int64(42), "name": "x"}
	if got := primaryKeyOf(row); got != int64(42) {
		t.Fatalf("primaryKeyOf: want 42, got %v", got)
	}
}

func TestPrimaryKeyOf_Missing(t *testing.T) {
	if got := primaryKeyOf(map[string]any{"name": "x"}); got != "<unknown>" {
		t.Fatalf("missing id: want <unknown>, got %v", got)
	}
}

// ─── DefaultConfig ────────────────────────────────────────────────────────────

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.BatchSize != 0 {
		t.Errorf("BatchSize: want 0, got %d", cfg.BatchSize)
	}
	if cfg.MaxRejects != 10 {
		t.Errorf("MaxRejects: want 10, got %d", cfg.MaxRejects)
	}
	if cfg.IgnoreIsFatal {
		t.Error("IgnoreIsFatal should default false")
	}
	if cfg.DryRun {
		t.Error("DryRun should default false")
	}
	if cfg.ProcessingTimeTarget != 20.0 {
		t.Errorf("ProcessingTimeTarget: want 20.0, got %v", cfg.ProcessingTimeTarget)
	}
	if cfg.CommitTimeTarget != 1.0 {
		t.Errorf("CommitTimeTarget: want 1.0, got %v", cfg.CommitTimeTarget)
	}
}

// ─── compileScript ────────────────────────────────────────────────────────────

func TestCompileScript_Valid(t *testing.T) {
	if _, err := compileScript(`x := 1 + 2`, nil); err != nil {
		t.Fatalf("valid script should compile: %v", err)
	}
}

func TestCompileScript_SyntaxError(t *testing.T) {
	if _, err := compileScript(`x := !!!`, nil); err == nil {
		t.Fatal("invalid script should fail to compile")
	}
}

func TestCompileScript_WithBuiltin(t *testing.T) {
	builtins := map[string]tengo.Object{
		"myFunc": &tengo.UserFunction{
			Name: "myFunc",
			Value: func(args ...tengo.Object) (tengo.Object, error) {
				return &tengo.Int{Value: 99}, nil
			},
		},
	}
	compiled, err := compileScript(`result := myFunc()`, builtins)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	if err := compiled.Run(); err != nil {
		t.Fatalf("run: %v", err)
	}
	v := compiled.Get("result")
	if i, ok := v.Object().(*tengo.Int); !ok || i.Value != 99 {
		t.Errorf("builtin result: want 99, got %v", v.Object())
	}
}

// ─── buildFieldList ───────────────────────────────────────────────────────────

func TestBuildFieldList(t *testing.T) {
	star := func(got []string) {
		t.Helper()
		if len(got) != 1 || got[0] != "*" {
			t.Errorf("want [*], got %v", got)
		}
	}
	star(buildFieldList(nil))
	star(buildFieldList(tengo.UndefinedValue))
	star(buildFieldList(&tengo.String{Value: ""}))
	star(buildFieldList(&tengo.Array{}))

	if got := buildFieldList(&tengo.String{Value: "name"}); len(got) != 1 || got[0] != "name" {
		t.Errorf("single string field: want [name], got %v", got)
	}

	arr := &tengo.Array{Value: []tengo.Object{
		&tengo.String{Value: "id"},
		&tengo.String{Value: "score"},
	}}
	got := buildFieldList(arr)
	if len(got) != 2 || got[0] != "id" || got[1] != "score" {
		t.Errorf("array fields: want [id score], got %v", got)
	}
}

// ─── singleColumnDef / buildColumnDefs ───────────────────────────────────────

func makeCM(kv ...string) *tengo.Map {
	m := &tengo.Map{Value: make(map[string]tengo.Object)}
	for i := 0; i+1 < len(kv); i += 2 {
		m.Value[kv[i]] = &tengo.String{Value: kv[i+1]}
	}
	return m
}

func setBoolKey(m *tengo.Map, key string, val bool) {
	if val {
		m.Value[key] = tengo.TrueValue
	} else {
		m.Value[key] = tengo.FalseValue
	}
}

func TestSingleColumnDef_Basic(t *testing.T) {
	m := makeCM("name", "id", "type", "SERIAL")
	def, err := singleColumnDef(m)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(def, "id") || !strings.Contains(def, "SERIAL") {
		t.Errorf("def missing name/type: %q", def)
	}
	// nullable defaults false → NOT NULL
	if !strings.Contains(def, "NOT NULL") {
		t.Errorf("nullable=false should produce NOT NULL: %q", def)
	}
}

func TestSingleColumnDef_AllOptions(t *testing.T) {
	m := makeCM("name", "email", "type", "TEXT", "default", "''")
	setBoolKey(m, "nullable", true)
	setBoolKey(m, "unique", true)
	setBoolKey(m, "primaryKey", true)
	def, err := singleColumnDef(m)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strings.Contains(def, "NOT NULL") {
		t.Errorf("nullable=true should omit NOT NULL: %q", def)
	}
	if !strings.Contains(def, "UNIQUE") {
		t.Errorf("unique=true should include UNIQUE: %q", def)
	}
	if !strings.Contains(def, "DEFAULT") {
		t.Errorf("default set should include DEFAULT: %q", def)
	}
	if !strings.Contains(def, "PRIMARY KEY") {
		t.Errorf("primaryKey=true should include PRIMARY KEY: %q", def)
	}
}

func TestSingleColumnDef_MissingName(t *testing.T) {
	if _, err := singleColumnDef(makeCM("type", "TEXT")); err == nil {
		t.Fatal("missing name should error")
	}
}

func TestSingleColumnDef_MissingType(t *testing.T) {
	if _, err := singleColumnDef(makeCM("name", "my_col")); err == nil {
		t.Fatal("missing type should error")
	}
}

func TestBuildColumnDefs_EmptyArray(t *testing.T) {
	defs, err := buildColumnDefs(&tengo.Array{})
	if err != nil {
		t.Fatalf("empty array should not error: %v", err)
	}
	if len(defs) != 0 {
		t.Errorf("empty array: want 0 defs, got %d", len(defs))
	}
}

func TestBuildColumnDefs_NonMapElement(t *testing.T) {
	arr := &tengo.Array{Value: []tengo.Object{&tengo.String{Value: "bad"}}}
	if _, err := buildColumnDefs(arr); err == nil {
		t.Fatal("non-map element should error")
	}
}

// ─── errMap ───────────────────────────────────────────────────────────────────

func TestErrMap(t *testing.T) {
	m := errMap(fmt.Errorf("something broke"))
	errVal, ok := m.Value["error"]
	if !ok {
		t.Fatal("errMap missing 'error' key")
	}
	s, ok := errVal.(*tengo.String)
	if !ok || s.Value != "something broke" {
		t.Fatalf("errMap value: want 'something broke', got %v", errVal)
	}
}

// ─── parseSelectArgs ──────────────────────────────────────────────────────────

func TestParseSelectArgs_Valid(t *testing.T) {
	args := []tengo.Object{
		&tengo.String{Value: "users"},
		&tengo.String{Value: "id > 0"},
		&tengo.String{Value: "name"},
	}
	table, where, fields, err := parseSelectArgs("select", args)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if table != "users" {
		t.Errorf("table: want users, got %q", table)
	}
	if where == nil || fields == nil {
		t.Error("where and fields should be populated")
	}
}

func TestParseSelectArgs_TableOnly(t *testing.T) {
	table, where, fields, err := parseSelectArgs("select", []tengo.Object{
		&tengo.String{Value: "tasks"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if table != "tasks" {
		t.Errorf("table: want tasks, got %q", table)
	}
	if where != nil || fields != nil {
		t.Error("where and fields should be nil for single arg")
	}
}

func TestParseSelectArgs_TooManyArgs(t *testing.T) {
	args := []tengo.Object{
		&tengo.String{Value: "t"},
		&tengo.String{Value: "w"},
		&tengo.String{Value: "f"},
		&tengo.String{Value: "extra"},
	}
	if _, _, _, err := parseSelectArgs("select", args); err == nil {
		t.Fatal("4 args should error")
	}
}

func TestParseSelectArgs_NoArgs(t *testing.T) {
	if _, _, _, err := parseSelectArgs("select", nil); err == nil {
		t.Fatal("0 args should error")
	}
}

func TestParseSelectArgs_NonStringTable(t *testing.T) {
	if _, _, _, err := parseSelectArgs("select", []tengo.Object{
		&tengo.Int{Value: 1},
	}); err == nil {
		t.Fatal("non-string table should error")
	}
}

// ─── applyWhere ───────────────────────────────────────────────────────────────

func TestApplyWhere_NilNoFilter(t *testing.T) {
	// Should not panic and should produce valid SQL without WHERE.
	sel := applyWhere(squirrelSelect(), nil)
	sql, _, err := sel.ToSql()
	if err != nil {
		t.Fatalf("ToSql: %v", err)
	}
	if strings.Contains(strings.ToUpper(sql), "WHERE") {
		t.Errorf("nil where: unexpected WHERE clause in %q", sql)
	}
}

func TestApplyWhere_UndefinedNoFilter(t *testing.T) {
	sel := applyWhere(squirrelSelect(), tengo.UndefinedValue)
	sql, _, err := sel.ToSql()
	if err != nil {
		t.Fatalf("ToSql: %v", err)
	}
	if strings.Contains(strings.ToUpper(sql), "WHERE") {
		t.Errorf("undefined where: unexpected WHERE clause in %q", sql)
	}
}

func TestApplyWhere_StringClause(t *testing.T) {
	sel := applyWhere(squirrelSelect(), &tengo.String{Value: "id > 5"})
	sql, _, err := sel.ToSql()
	if err != nil {
		t.Fatalf("ToSql: %v", err)
	}
	if !strings.Contains(sql, "id > 5") {
		t.Errorf("string clause not in SQL: %q", sql)
	}
}

func TestApplyWhere_MapClause(t *testing.T) {
	m := &tengo.Map{Value: map[string]tengo.Object{
		"status": &tengo.String{Value: "active"},
	}}
	sel := applyWhere(squirrelSelect(), m)
	sql, _, err := sel.ToSql()
	if err != nil {
		t.Fatalf("ToSql: %v", err)
	}
	if !strings.Contains(sql, "status") {
		t.Errorf("map clause 'status' not in SQL: %q", sql)
	}
}

// ─── Record built-in functions (no DB) ───────────────────────────────────────

func TestBuiltin_GetField(t *testing.T) {
	rec := NewRecord(map[string]any{"score": int64(5)})
	compiled := runTengo(t, `v := getField("score")`, noDBBuiltins(rec))
	v := compiled.Get("v")
	if i, ok := v.Object().(*tengo.Int); !ok || i.Value != 5 {
		t.Fatalf("getField: want 5, got %v", v.Object())
	}
}

func TestBuiltin_SetField(t *testing.T) {
	rec := NewRecord(map[string]any{"score": int64(5)})
	runTengo(t, `setField("score", 99)`, noDBBuiltins(rec))
	if rec.Get("score") != int64(99) {
		t.Fatalf("setField: want 99, got %v", rec.Get("score"))
	}
}

func TestBuiltin_DeleteField(t *testing.T) {
	rec := NewRecord(map[string]any{"tmp": "remove_me"})
	runTengo(t, `deleteField("tmp")`, noDBBuiltins(rec))
	if _, ok := rec.MergedRow()["tmp"]; ok {
		t.Fatal("deleteField: field should be absent from MergedRow")
	}
}

func TestBuiltin_RejectRecord(t *testing.T) {
	rec := NewRecord(nil)
	runTengo(t, `rejectRecord("bad record")`, noDBBuiltins(rec))
	if !rec.rejected || rec.rejectMsg != "bad record" {
		t.Fatalf("rejectRecord: rejected=%v msg=%q", rec.rejected, rec.rejectMsg)
	}
}

func TestBuiltin_RejectRecord_DefaultMsg(t *testing.T) {
	rec := NewRecord(nil)
	runTengo(t, `rejectRecord()`, noDBBuiltins(rec))
	if !rec.rejected || rec.rejectMsg == "" {
		t.Fatal("rejectRecord with no arg should set a non-empty default message")
	}
}

func TestBuiltin_IgnoreRecord(t *testing.T) {
	rec := NewRecord(nil)
	runTengo(t, `ignoreRecord("skip")`, noDBBuiltins(rec))
	if !rec.ignored || rec.ignoreMsg != "skip" {
		t.Fatalf("ignoreRecord: ignored=%v msg=%q", rec.ignored, rec.ignoreMsg)
	}
}

func TestBuiltin_IgnoreRecord_DefaultMsg(t *testing.T) {
	rec := NewRecord(nil)
	runTengo(t, `ignoreRecord()`, noDBBuiltins(rec))
	if !rec.ignored || rec.ignoreMsg == "" {
		t.Fatal("ignoreRecord with no arg should set a non-empty default message")
	}
}

func TestBuiltin_GetField_WrongArgCount(t *testing.T) {
	rec := NewRecord(nil)
	compiled, err := compileScript(`getField()`, noDBBuiltins(rec))
	if err != nil {
		return // compile-time error is fine
	}
	if err := compiled.Run(); err == nil {
		t.Fatal("getField() with 0 args should produce a runtime error")
	}
}

func TestBuiltin_SetField_WrongArgCount(t *testing.T) {
	rec := NewRecord(nil)
	compiled, err := compileScript(`setField("x")`, noDBBuiltins(rec))
	if err != nil {
		return
	}
	if err := compiled.Run(); err == nil {
		t.Fatal("setField with 1 arg should produce a runtime error")
	}
}

// ─── hash built-in ────────────────────────────────────────────────────────────

func TestBuiltin_Hash_SHA256(t *testing.T) {
	rec := NewRecord(nil)
	compiled := runTengo(t, `result := hash("sha256", "hello")`, noDBBuiltins(rec))
	got := getStr(t, compiled, "result")
	// SHA-256("hello") is deterministic.
	want := "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
	if got != want {
		t.Errorf("sha256(hello) = %q, want %q", got, want)
	}
}

func TestBuiltin_Hash_SHA512(t *testing.T) {
	rec := NewRecord(nil)
	compiled := runTengo(t, `result := hash("sha512", "")`, noDBBuiltins(rec))
	got := getStr(t, compiled, "result")
	if len(got) != 128 {
		t.Errorf("sha512 hex length: want 128, got %d", len(got))
	}
}

func TestBuiltin_Hash_MD5(t *testing.T) {
	rec := NewRecord(nil)
	compiled := runTengo(t, `result := hash("md5", "test")`, noDBBuiltins(rec))
	got := getStr(t, compiled, "result")
	if len(got) != 32 {
		t.Errorf("md5 hex length: want 32, got %d", len(got))
	}
}

func TestBuiltin_Hash_Blake2b(t *testing.T) {
	rec := NewRecord(nil)
	compiled := runTengo(t, `result := hash("blake2b", "data")`, noDBBuiltins(rec))
	got := getStr(t, compiled, "result")
	if len(got) == 0 {
		t.Error("blake2b: expected non-empty result")
	}
}

func TestBuiltin_Hash_UnknownMethod(t *testing.T) {
	rec := NewRecord(nil)
	compiled := runTengo(t, `result := hash("nope", "data")`, noDBBuiltins(rec))
	got := getStr(t, compiled, "result")
	if !strings.Contains(got, "unknown method") {
		t.Errorf("unknown method should return error string, got %q", got)
	}
}

func TestBuiltin_Hash_WrongArgCount(t *testing.T) {
	rec := NewRecord(nil)
	compiled, err := compileScript(`hash("sha256")`, noDBBuiltins(rec))
	if err != nil {
		return
	}
	if err := compiled.Run(); err == nil {
		t.Fatal("hash with 1 arg should produce a runtime error")
	}
}

// ─── uuid_5 built-in ──────────────────────────────────────────────────────────

func TestBuiltin_UUID5_Valid(t *testing.T) {
	rec := NewRecord(nil)
	// Use DNS namespace UUID (RFC 4122).
	compiled := runTengo(t,
		`result := uuid_5("6ba7b810-9dad-11d1-80b4-00c04fd430c8", "example.com")`,
		noDBBuiltins(rec),
	)
	got := getStr(t, compiled, "result")
	if len(got) != 36 || strings.Count(got, "-") != 4 {
		t.Errorf("uuid_5 result not UUID-shaped: %q", got)
	}
}

func TestBuiltin_UUID5_Deterministic(t *testing.T) {
	// Same inputs → same UUID.
	script := `result := uuid_5("6ba7b810-9dad-11d1-80b4-00c04fd430c8", "same")`
	rec1 := NewRecord(nil)
	c1 := runTengo(t, script, noDBBuiltins(rec1))
	rec2 := NewRecord(nil)
	c2 := runTengo(t, script, noDBBuiltins(rec2))
	if getStr(t, c1, "result") != getStr(t, c2, "result") {
		t.Error("uuid_5 should be deterministic for identical inputs")
	}
}

func TestBuiltin_UUID5_InvalidNamespace(t *testing.T) {
	rec := NewRecord(nil)
	compiled := runTengo(t, `result := uuid_5("not-a-uuid", "name")`, noDBBuiltins(rec))
	got := getStr(t, compiled, "result")
	if !strings.Contains(got, "invalid") {
		t.Errorf("invalid namespace should return error string, got %q", got)
	}
}

// ─── now built-in ─────────────────────────────────────────────────────────────

func TestBuiltin_Now(t *testing.T) {
	rec := NewRecord(nil)
	compiled := runTengo(t, `
		t := now()
		u := t.unix()
		r := t.rfc3339()
		y := t.format("2006")
	`, noDBBuiltins(rec))

	// unix() → positive integer
	uv := compiled.Get("u")
	ui, ok := uv.Object().(*tengo.Int)
	if !ok || ui.Value <= 0 {
		t.Fatalf("now().unix(): want positive Int, got %v", uv.Object())
	}
	// rfc3339() → contains "T"
	rv := getStr(t, compiled, "r")
	if !strings.Contains(rv, "T") {
		t.Errorf("now().rfc3339() should contain 'T', got %q", rv)
	}
	// format("2006") → 4-char year
	yv := getStr(t, compiled, "y")
	if len(yv) != 4 {
		t.Errorf("now().format('2006') should be 4-char year, got %q", yv)
	}
}

func TestBuiltin_Now_FormatWrongArgs(t *testing.T) {
	rec := NewRecord(nil)
	compiled, err := compileScript(`t := now(); t.format()`, noDBBuiltins(rec))
	if err != nil {
		return
	}
	if err := compiled.Run(); err == nil {
		t.Fatal("now().format() with 0 args should produce a runtime error")
	}
}

// ─── field built-in ───────────────────────────────────────────────────────────

func TestBuiltin_Field(t *testing.T) {
	rec := NewRecord(nil)
	compiled := runTengo(t, `ref := field("my_col")`, noDBBuiltins(rec))
	refObj := compiled.Get("ref")
	m, ok := refObj.Object().(*tengo.Map)
	if !ok {
		t.Fatalf("field() should return *tengo.Map, got %T", refObj.Object())
	}
	if _, ok := m.Value["__field_ref__"]; !ok {
		t.Error("field() map should have '__field_ref__' key")
	}
}

// ─── Runner invalid batch-size -1 ────────────────────────────────────────────

func TestRunner_InvalidBatchSize(t *testing.T) {
	// NewRunner with BatchSize=-1 should return error from Run without touching DB.
	r := NewRunner(nil, Config{BatchSize: -1, MaxRejects: 10})
	err := r.Run(nil, 1) //nolint:staticcheck // nil ctx intentional (error before DB use)
	if err == nil || !strings.Contains(err.Error(), "batch-size -1") {
		t.Fatalf("BatchSize=-1 should produce descriptive error, got %v", err)
	}
}
