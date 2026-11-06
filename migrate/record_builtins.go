package migrate

// record_builtins.go — Tengo built-ins for record_migration_script (per-record context).
// Design: bakeoff#27 Phase 1 sign-off (050558ZJUN26).
//
// Each built-in operates on the current Record (rec) and the DB connection.
// record disposition (reject/ignore) is tracked on the Record struct itself
// and read by the runner after script execution completes.

import (
	"context"
	"crypto/md5" //nolint:gosec // backs the script-selectable md5 hash method, not a security primitive
	"crypto/sha256"
	"crypto/sha512"
	"fmt"
	"strings"
	"time"

	"github.com/Masterminds/squirrel"
	"github.com/d5/tengo/v2"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"golang.org/x/crypto/blake2b"
)

// recordEnv carries what the record built-ins close over: the mutable current
// record and the pgx connection.
type recordEnv struct {
	conn *pgx.Conn
	rec  *Record
	psql squirrel.StatementBuilderType
}

func userFunc(name string, fn tengo.CallableFunc) *tengo.UserFunction {
	return &tengo.UserFunction{Name: name, Value: fn}
}

// recordBuiltins returns the Tengo built-in map for record_migration_script execution.
// rec is the mutable current record; conn is the pgx connection.
func recordBuiltins(ctx context.Context, conn *pgx.Conn, rec *Record) map[string]tengo.Object {
	env := &recordEnv{conn: conn, rec: rec, psql: squirrel.StatementBuilder.PlaceholderFormat(squirrel.Dollar)}

	return map[string]tengo.Object{
		// getField(field) — returns this[field]
		"getField": userFunc("getField", env.getField()),
		// setField(field, value) — sets this[field], marks dirty
		"setField": userFunc("setField", env.setField()),
		// deleteField(field) — adds to delete-set; omit on write-back
		"deleteField": userFunc("deleteField", env.deleteField()),
		// rejectRecord(reason) — per-record abort, counted against --max-rejects
		"rejectRecord": userFunc("rejectRecord", env.reasoned("rejected by migration script", rec.Reject)),
		// ignoreRecord(reason) — per-record skip
		"ignoreRecord": userFunc("ignoreRecord", env.reasoned("ignored by migration script", rec.Ignore)),
		// select(table, where, fields) — squirrel SELECT; returns []map
		"select": userFunc("select", env.selectRows(ctx)),
		// selectFirst(table, where, fields) — LIMIT 1
		"selectFirst": userFunc("selectFirst", env.selectFirst(ctx)),
		// selectOne(table, where, fields) — LIMIT 2; error if count != 1
		"selectOne": userFunc("selectOne", env.selectOne(ctx)),
		// rawSQL(stmt) — restored for Phase 1 per bakeoff#27 (050429ZJUN26).
		// Removal gate: all DB records confirmed clean + NOMAD approval.
		"rawSQL": userFunc("rawSQL", env.rawSQL(ctx)),
		// hash(method, content) — sha256/sha512/md5/blake2b via Go crypto
		"hash": userFunc("hash", hashBuiltin()),
		// uuid_5(namespace, name) — UUID v5
		"uuid_5": userFunc("uuid_5", uuid5Builtin()),
		// now() — returns a TimeObject map with .unix(), .rfc3339(), .format(layout)
		"now": userFunc("now", nowBuiltin()),
		// field(name) — alias for squirrel.Expr(), used in where clauses
		"field": userFunc("field", fieldBuiltin()),
	}
}

func (e *recordEnv) getField() tengo.CallableFunc {
	return func(args ...tengo.Object) (tengo.Object, error) {
		if len(args) != 1 {
			return nil, fmt.Errorf("getField expects 1 arg")
		}
		field, ok := args[0].(*tengo.String)
		if !ok {
			return nil, fmt.Errorf("getField: field must be string")
		}
		return goToTengo(e.rec.Get(field.Value)), nil
	}
}

func (e *recordEnv) setField() tengo.CallableFunc {
	return func(args ...tengo.Object) (tengo.Object, error) {
		if len(args) != 2 {
			return nil, fmt.Errorf("setField expects 2 args")
		}
		field, ok := args[0].(*tengo.String)
		if !ok {
			return nil, fmt.Errorf("setField: field must be string")
		}
		e.rec.Set(field.Value, tengoToGo(args[1]))
		return tengo.UndefinedValue, nil
	}
}

func (e *recordEnv) deleteField() tengo.CallableFunc {
	return func(args ...tengo.Object) (tengo.Object, error) {
		if len(args) != 1 {
			return nil, fmt.Errorf("deleteField expects 1 arg")
		}
		field, ok := args[0].(*tengo.String)
		if !ok {
			return nil, fmt.Errorf("deleteField: field must be string")
		}
		e.rec.Delete(field.Value)
		return tengo.UndefinedValue, nil
	}
}

// reasoned builds a built-in that passes its optional string argument to apply,
// falling back to defaultReason.
func (e *recordEnv) reasoned(defaultReason string, apply func(string)) tengo.CallableFunc {
	return func(args ...tengo.Object) (tengo.Object, error) {
		reason := defaultReason
		if len(args) > 0 {
			if s, ok := args[0].(*tengo.String); ok {
				reason = s.Value
			}
		}
		apply(reason)
		return tengo.UndefinedValue, nil
	}
}

// queryRows executes a squirrel select and returns []map[string]any.
func (e *recordEnv) queryRows(ctx context.Context, table string, where, fields tengo.Object) ([]map[string]any, error) {
	sel := e.psql.Select(buildFieldList(fields)...).From(table)
	sel = applyWhere(sel, where)
	sql, args, err := sel.ToSql()
	if err != nil {
		return nil, fmt.Errorf("select: build query: %w", err)
	}
	rows, err := e.conn.Query(ctx, sql, args...)
	if err != nil {
		return nil, fmt.Errorf("select: %w", err)
	}
	defer rows.Close()
	return pgx.CollectRows(rows, pgx.RowToMap)
}

// queryLimited runs a select of at most limit rows for selectFirst/selectOne.
// Query-build and query errors are returned wrapped with fn; collection errors are not.
func (e *recordEnv) queryLimited(ctx context.Context, fn string, limit uint64, args []tengo.Object) ([]map[string]any, error, error) {
	table, where, fields, err := parseSelectArgs(fn, args)
	if err != nil {
		return nil, nil, err
	}
	sel := e.psql.Select(buildFieldList(fields)...).From(table).Limit(limit)
	sel = applyWhere(sel, where)
	sql, qArgs, err := sel.ToSql()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", fn, err), nil
	}
	pgRows, err := e.conn.Query(ctx, sql, qArgs...)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", fn, err), nil
	}
	defer pgRows.Close()
	results, err := pgx.CollectRows(pgRows, pgx.RowToMap)
	return results, err, nil
}

func (e *recordEnv) selectRows(ctx context.Context) tengo.CallableFunc {
	return func(args ...tengo.Object) (tengo.Object, error) {
		table, where, fields, err := parseSelectArgs("select", args)
		if err != nil {
			return nil, err
		}
		rows, err := e.queryRows(ctx, table, where, fields)
		if err != nil {
			return errMap(err), nil
		}
		return rowsToTengo(rows), nil
	}
}

func (e *recordEnv) selectFirst(ctx context.Context) tengo.CallableFunc {
	return func(args ...tengo.Object) (tengo.Object, error) {
		results, scriptErr, argErr := e.queryLimited(ctx, "selectFirst", 1, args)
		if argErr != nil {
			return nil, argErr
		}
		if scriptErr != nil {
			return errMap(scriptErr), nil
		}
		if len(results) == 0 {
			return tengo.UndefinedValue, nil
		}
		return rowToTengo(results[0]), nil
	}
}

func (e *recordEnv) selectOne(ctx context.Context) tengo.CallableFunc {
	return func(args ...tengo.Object) (tengo.Object, error) {
		results, scriptErr, argErr := e.queryLimited(ctx, "selectOne", 2, args)
		if argErr != nil {
			return nil, argErr
		}
		if scriptErr != nil {
			return errMap(scriptErr), nil
		}
		if len(results) != 1 {
			return errMap(fmt.Errorf("selectOne: expected exactly 1 row, got %d", len(results))), nil
		}
		return rowToTengo(results[0]), nil
	}
}

func (e *recordEnv) rawSQL(ctx context.Context) tengo.CallableFunc {
	return func(args ...tengo.Object) (tengo.Object, error) {
		if len(args) != 1 {
			return nil, fmt.Errorf("rawSQL expects 1 arg")
		}
		stmt, ok := args[0].(*tengo.String)
		if !ok {
			return nil, fmt.Errorf("rawSQL: stmt must be string")
		}
		rows, err := e.conn.Query(ctx, stmt.Value)
		if err != nil {
			return errMap(err), nil
		}
		defer rows.Close()
		results, err := pgx.CollectRows(rows, pgx.RowToMap)
		if err != nil {
			return errMap(err), nil
		}
		return rowsToTengo(results), nil
	}
}

func hashBuiltin() tengo.CallableFunc {
	return func(args ...tengo.Object) (tengo.Object, error) {
		if len(args) != 2 {
			return nil, fmt.Errorf("hash expects 2 args (method, content)")
		}
		method, ok1 := args[0].(*tengo.String)
		content, ok2 := args[1].(*tengo.String)
		if !ok1 || !ok2 {
			return nil, fmt.Errorf("hash: both args must be strings")
		}
		data := []byte(content.Value)
		var result string
		switch strings.ToLower(method.Value) {
		case "sha256":
			h := sha256.Sum256(data)
			result = fmt.Sprintf("%x", h[:])
		case "sha512":
			h := sha512.Sum512(data)
			result = fmt.Sprintf("%x", h[:])
		case "md5":
			h := md5.Sum(data) //nolint:gosec // script-selected md5 checksum for content comparison, not a security primitive
			result = fmt.Sprintf("%x", h[:])
		case "blake2b":
			h, err := blake2b.New256(nil)
			if err != nil {
				return nil, fmt.Errorf("hash: blake2b init: %w", err)
			}
			h.Write(data)
			result = fmt.Sprintf("%x", h.Sum(nil))
		default:
			return &tengo.String{Value: fmt.Sprintf("hash: unknown method %q", method.Value)}, nil
		}
		return &tengo.String{Value: result}, nil
	}
}

func uuid5Builtin() tengo.CallableFunc {
	return func(args ...tengo.Object) (tengo.Object, error) {
		if len(args) != 2 {
			return nil, fmt.Errorf("uuid_5 expects 2 args (namespace, name)")
		}
		ns, ok1 := args[0].(*tengo.String)
		name, ok2 := args[1].(*tengo.String)
		if !ok1 || !ok2 {
			return nil, fmt.Errorf("uuid_5: both args must be strings")
		}
		nsUUID, err := uuid.Parse(ns.Value)
		if err != nil {
			return &tengo.String{Value: fmt.Sprintf("uuid_5: invalid namespace UUID: %v", err)}, nil
		}
		result := uuid.NewSHA1(nsUUID, []byte(name.Value))
		return &tengo.String{Value: result.String()}, nil
	}
}

func nowBuiltin() tengo.CallableFunc {
	return func(_ ...tengo.Object) (tengo.Object, error) {
		t := time.Now().UTC()
		return &tengo.Map{Value: map[string]tengo.Object{
			"unix": userFunc("unix", func(_ ...tengo.Object) (tengo.Object, error) {
				return &tengo.Int{Value: t.Unix()}, nil
			}),
			"rfc3339": userFunc("rfc3339", func(_ ...tengo.Object) (tengo.Object, error) {
				return &tengo.String{Value: t.Format(time.RFC3339)}, nil
			}),
			"format": userFunc("format", func(layoutArgs ...tengo.Object) (tengo.Object, error) {
				if len(layoutArgs) != 1 {
					return nil, fmt.Errorf("now().format expects 1 arg (layout)")
				}
				layout, ok := layoutArgs[0].(*tengo.String)
				if !ok {
					return nil, fmt.Errorf("now().format: layout must be string")
				}
				return &tengo.String{Value: t.Format(layout.Value)}, nil
			}),
		}}, nil
	}
}

func fieldBuiltin() tengo.CallableFunc {
	return func(args ...tengo.Object) (tengo.Object, error) {
		if len(args) != 1 {
			return nil, fmt.Errorf("field expects 1 arg")
		}
		name, ok := args[0].(*tengo.String)
		if !ok {
			return nil, fmt.Errorf("field: arg must be string")
		}
		// Return a special marker map that applyWhere recognises as a column ref.
		return &tengo.Map{Value: map[string]tengo.Object{
			"__field_ref__": &tengo.String{Value: name.Value},
		}}, nil
	}
}

// parseSelectArgs validates and extracts (table, where, fields) from select* args.
func parseSelectArgs(fn string, args []tengo.Object) (string, tengo.Object, tengo.Object, error) { //nolint:ireturn // tengo.Object is a third-party interface (d5/tengo); where/fields pass through whatever concrete arg type the caller supplied.
	if len(args) < 1 || len(args) > 3 {
		return "", nil, nil, fmt.Errorf("%s expects 1–3 args (table, [where], [fields])", fn)
	}
	tableObj, ok := args[0].(*tengo.String)
	if !ok {
		return "", nil, nil, fmt.Errorf("%s: table must be string", fn)
	}
	var where, fields tengo.Object
	if len(args) >= 2 {
		where = args[1]
	}
	if len(args) == 3 {
		fields = args[2]
	}
	return tableObj.Value, where, fields, nil
}

// buildFieldList extracts a SQL field list from a Tengo value.
// nil / undefined / empty array → ["*"].
func buildFieldList(fields tengo.Object) []string {
	if fields == nil || fields == tengo.UndefinedValue {
		return []string{"*"}
	}
	switch v := fields.(type) {
	case *tengo.String:
		if v.Value == "" {
			return []string{"*"}
		}
		return []string{v.Value}
	case *tengo.Array:
		if len(v.Value) == 0 {
			return []string{"*"}
		}
		out := make([]string, 0, len(v.Value))
		for _, elem := range v.Value {
			if s, ok := elem.(*tengo.String); ok {
				out = append(out, s.Value)
			}
		}
		if len(out) == 0 {
			return []string{"*"}
		}
		return out
	}
	return []string{"*"}
}

// applyWhere applies a Tengo where clause to a squirrel SelectBuilder.
// where can be:
//   - nil/undefined: no filter
//   - *tengo.String: raw SQL expression
//   - *tengo.Map: key→value pairs (implicit AND); values may be squirrel operators
func applyWhere(sel squirrel.SelectBuilder, where tengo.Object) squirrel.SelectBuilder {
	if where == nil || where == tengo.UndefinedValue {
		return sel
	}
	switch w := where.(type) {
	case *tengo.String:
		return sel.Where(w.Value)
	case *tengo.Map:
		eq := buildEqMap(w)
		return sel.Where(eq)
	}
	return sel
}

// buildEqMap converts a Tengo map to a squirrel Eq clause.
// Supports squirrel operator wrappers stored as {__op__: op, __val__: val} maps.
func buildEqMap(m *tengo.Map) squirrel.Eq {
	eq := squirrel.Eq{}
	for k, v := range m.Value {
		eq[k] = tengoToGo(v)
	}
	return eq
}

// rowToTengo converts a DB row map to a Tengo map.
func rowToTengo(row map[string]any) *tengo.Map {
	m := &tengo.Map{Value: make(map[string]tengo.Object, len(row))}
	for k, v := range row {
		m.Value[k] = goToTengo(v)
	}
	return m
}

// rowsToTengo converts a slice of row maps to a Tengo array.
func rowsToTengo(rows []map[string]any) *tengo.Array {
	arr := &tengo.Array{Value: make([]tengo.Object, len(rows))}
	for i, row := range rows {
		arr.Value[i] = rowToTengo(row)
	}
	return arr
}

// errMap wraps an error into a Tengo map with an "error" key.
// Scripts should check: if result.error { rejectRecord(result.error) }.
func errMap(err error) *tengo.Map {
	return &tengo.Map{Value: map[string]tengo.Object{
		"error": &tengo.String{Value: err.Error()},
	}}
}
