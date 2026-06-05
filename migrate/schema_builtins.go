package migrate

// schema_builtins.go — Tengo built-ins for schema_migration_script (DDL context).
// Design: bakeoff#27 Phase 1 sign-off (050558ZJUN26).
//
// Each built-in executes SQL against the database immediately and returns
// an empty string on success or an error string on failure.
// Scripts should check: if err := createTable(...); err != "" { fail(err) }

import (
	"context"
	"fmt"
	"strings"

	"github.com/Masterminds/squirrel"
	"github.com/d5/tengo/v2"
	"github.com/jackc/pgx/v5"
)

// schemaBuiltins returns the Tengo built-in map for schema_migration_script execution.
// conn is the live pgx connection for DDL execution.
func schemaBuiltins(ctx context.Context, conn *pgx.Conn) map[string]tengo.Object {
	exec := func(sql string, args ...any) string {
		_, err := conn.Exec(ctx, sql, args...)
		if err != nil {
			return err.Error()
		}
		return ""
	}

	return map[string]tengo.Object{
		// createTable(name, cols)
		// cols is an array of maps: {name, type, nullable, default, unique, primaryKey}
		"createTable": &tengo.UserFunction{
			Name: "createTable",
			Value: func(args ...tengo.Object) (tengo.Object, error) {
				if len(args) != 2 {
					return nil, fmt.Errorf("createTable expects 2 args")
				}
				name, ok := args[0].(*tengo.String)
				if !ok {
					return nil, fmt.Errorf("createTable: name must be string")
				}
				colsArr, ok := args[1].(*tengo.Array)
				if !ok {
					return nil, fmt.Errorf("createTable: cols must be array")
				}
				colDefs, err := buildColumnDefs(colsArr)
				if err != nil {
					return &tengo.String{Value: err.Error()}, nil
				}
				sql := fmt.Sprintf("CREATE TABLE IF NOT EXISTS %s (%s)", name.Value, strings.Join(colDefs, ", "))
				return &tengo.String{Value: exec(sql)}, nil
			},
		},

		// dropTable(name)
		"dropTable": &tengo.UserFunction{
			Name: "dropTable",
			Value: func(args ...tengo.Object) (tengo.Object, error) {
				if len(args) != 1 {
					return nil, fmt.Errorf("dropTable expects 1 arg")
				}
				name, ok := args[0].(*tengo.String)
				if !ok {
					return nil, fmt.Errorf("dropTable: name must be string")
				}
				return &tengo.String{Value: exec(fmt.Sprintf("DROP TABLE IF EXISTS %s", name.Value))}, nil
			},
		},

		// renameTable(old, new)
		"renameTable": &tengo.UserFunction{
			Name: "renameTable",
			Value: func(args ...tengo.Object) (tengo.Object, error) {
				if len(args) != 2 {
					return nil, fmt.Errorf("renameTable expects 2 args")
				}
				oldName, ok1 := args[0].(*tengo.String)
				newName, ok2 := args[1].(*tengo.String)
				if !ok1 || !ok2 {
					return nil, fmt.Errorf("renameTable: both args must be strings")
				}
				return &tengo.String{Value: exec(fmt.Sprintf("ALTER TABLE %s RENAME TO %s", oldName.Value, newName.Value))}, nil
			},
		},

		// addColumn(table, col)
		// col is a map: {name, type, nullable, default, unique, primaryKey}
		"addColumn": &tengo.UserFunction{
			Name: "addColumn",
			Value: func(args ...tengo.Object) (tengo.Object, error) {
				if len(args) != 2 {
					return nil, fmt.Errorf("addColumn expects 2 args")
				}
				table, ok := args[0].(*tengo.String)
				if !ok {
					return nil, fmt.Errorf("addColumn: table must be string")
				}
				colMap, ok := args[1].(*tengo.Map)
				if !ok {
					return nil, fmt.Errorf("addColumn: col must be map")
				}
				colDef, err := singleColumnDef(colMap)
				if err != nil {
					return &tengo.String{Value: err.Error()}, nil
				}
				return &tengo.String{Value: exec(fmt.Sprintf("ALTER TABLE %s ADD COLUMN IF NOT EXISTS %s", table.Value, colDef))}, nil
			},
		},

		// dropColumn(table, col)
		"dropColumn": &tengo.UserFunction{
			Name: "dropColumn",
			Value: func(args ...tengo.Object) (tengo.Object, error) {
				if len(args) != 2 {
					return nil, fmt.Errorf("dropColumn expects 2 args")
				}
				table, ok1 := args[0].(*tengo.String)
				col, ok2 := args[1].(*tengo.String)
				if !ok1 || !ok2 {
					return nil, fmt.Errorf("dropColumn: both args must be strings")
				}
				return &tengo.String{Value: exec(fmt.Sprintf("ALTER TABLE %s DROP COLUMN IF EXISTS %s", table.Value, col.Value))}, nil
			},
		},

		// renameColumn(table, old, new)
		"renameColumn": &tengo.UserFunction{
			Name: "renameColumn",
			Value: func(args ...tengo.Object) (tengo.Object, error) {
				if len(args) != 3 {
					return nil, fmt.Errorf("renameColumn expects 3 args")
				}
				table, ok1 := args[0].(*tengo.String)
				oldCol, ok2 := args[1].(*tengo.String)
				newCol, ok3 := args[2].(*tengo.String)
				if !ok1 || !ok2 || !ok3 {
					return nil, fmt.Errorf("renameColumn: all args must be strings")
				}
				return &tengo.String{Value: exec(fmt.Sprintf("ALTER TABLE %s RENAME COLUMN %s TO %s", table.Value, oldCol.Value, newCol.Value))}, nil
			},
		},

		// createIndex(table, name, cols, unique)
		// cols is an array of strings; unique is bool
		"createIndex": &tengo.UserFunction{
			Name: "createIndex",
			Value: func(args ...tengo.Object) (tengo.Object, error) {
				if len(args) != 4 {
					return nil, fmt.Errorf("createIndex expects 4 args (table, name, cols, unique)")
				}
				table, ok1 := args[0].(*tengo.String)
				name, ok2 := args[1].(*tengo.String)
				colsArr, ok3 := args[2].(*tengo.Array)
				if !ok1 || !ok2 || !ok3 {
					return nil, fmt.Errorf("createIndex: invalid arg types")
				}
				unique := false
				if b, ok := args[3].(*tengo.Bool); ok {
					unique = !b.IsFalsy()
				}
				cols := make([]string, 0, len(colsArr.Value))
				for _, c := range colsArr.Value {
					s, ok := c.(*tengo.String)
					if !ok {
						return nil, fmt.Errorf("createIndex: cols must be array of strings")
					}
					cols = append(cols, s.Value)
				}
				uniqueKw := ""
				if unique {
					uniqueKw = "UNIQUE "
				}
				sql := fmt.Sprintf("CREATE %sINDEX IF NOT EXISTS %s ON %s (%s)", uniqueKw, name.Value, table.Value, strings.Join(cols, ", "))
				return &tengo.String{Value: exec(sql)}, nil
			},
		},

		// dropIndex(name)
		"dropIndex": &tengo.UserFunction{
			Name: "dropIndex",
			Value: func(args ...tengo.Object) (tengo.Object, error) {
				if len(args) != 1 {
					return nil, fmt.Errorf("dropIndex expects 1 arg")
				}
				name, ok := args[0].(*tengo.String)
				if !ok {
					return nil, fmt.Errorf("dropIndex: name must be string")
				}
				return &tengo.String{Value: exec(fmt.Sprintf("DROP INDEX IF EXISTS %s", name.Value))}, nil
			},
		},

		// rawSQL(stmt)
		"rawSQL": &tengo.UserFunction{
			Name: "rawSQL",
			Value: func(args ...tengo.Object) (tengo.Object, error) {
				if len(args) != 1 {
					return nil, fmt.Errorf("rawSQL expects 1 arg")
				}
				stmt, ok := args[0].(*tengo.String)
				if !ok {
					return nil, fmt.Errorf("rawSQL: stmt must be string")
				}
				return &tengo.String{Value: exec(stmt.Value)}, nil
			},
		},

		// withIndexDisabled(table, indexName, block)
		// Fetches DDL from pg_indexes, drops the index, executes block, recreates index.
		// block must be a callable Tengo object (function or closure).
		"withIndexDisabled": &tengo.UserFunction{
			Name: "withIndexDisabled",
			Value: func(args ...tengo.Object) (tengo.Object, error) {
				if len(args) != 3 {
					return nil, fmt.Errorf("withIndexDisabled expects 3 args (table, indexName, block)")
				}
				table, ok1 := args[0].(*tengo.String)
				indexName, ok2 := args[1].(*tengo.String)
				if !ok1 || !ok2 {
					return nil, fmt.Errorf("withIndexDisabled: table and indexName must be strings")
				}
				block := args[2]
				if !block.CanCall() {
					return nil, fmt.Errorf("withIndexDisabled: block must be callable")
				}

				// 1. Fetch index DDL from pg_indexes
				var indexDDL string
				row := conn.QueryRow(ctx,
					"SELECT indexdef FROM pg_indexes WHERE tablename = $1 AND indexname = $2",
					table.Value, indexName.Value)
				if err := row.Scan(&indexDDL); err != nil {
					return &tengo.String{Value: fmt.Sprintf("withIndexDisabled: index not found: %v", err)}, nil
				}

				// 2. Drop index
				if errStr := exec(fmt.Sprintf("DROP INDEX IF EXISTS %s", indexName.Value)); errStr != "" {
					return &tengo.String{Value: errStr}, nil
				}

				// 3. Execute block
				if _, err := block.Call(); err != nil {
					return &tengo.String{Value: fmt.Sprintf("withIndexDisabled: block error: %v", err)}, nil
				}

				// 4. Recreate index
				return &tengo.String{Value: exec(indexDDL)}, nil
			},
		},

		// fail(msg) — abort script with error
		"fail": &tengo.UserFunction{
			Name: "fail",
			Value: func(args ...tengo.Object) (tengo.Object, error) {
				msg := "migration script called fail()"
				if len(args) > 0 {
					if s, ok := args[0].(*tengo.String); ok {
						msg = s.Value
					}
				}
				return nil, fmt.Errorf("%s", msg)
			},
		},
	}
}

// buildColumnDefs converts a Tengo array of column maps to SQL column definition strings.
func buildColumnDefs(colsArr *tengo.Array) ([]string, error) {
	defs := make([]string, 0, len(colsArr.Value))
	for _, elem := range colsArr.Value {
		colMap, ok := elem.(*tengo.Map)
		if !ok {
			return nil, fmt.Errorf("cols element must be a map")
		}
		def, err := singleColumnDef(colMap)
		if err != nil {
			return nil, err
		}
		defs = append(defs, def)
	}
	return defs, nil
}

// singleColumnDef converts a Tengo column map to a SQL column definition string.
// Expected keys: name (string), type (string), nullable (bool), default (string),
// unique (bool), primaryKey (bool).
func singleColumnDef(colMap *tengo.Map) (string, error) {
	getName := func(key string) string {
		if v, ok := colMap.Value[key]; ok {
			if s, ok := v.(*tengo.String); ok {
				return s.Value
			}
		}
		return ""
	}
	getBool := func(key string) bool {
		if v, ok := colMap.Value[key]; ok {
			return !v.IsFalsy()
		}
		return false
	}

	name := getName("name")
	typ := getName("type")
	if name == "" || typ == "" {
		return "", fmt.Errorf("column map missing required 'name' or 'type'")
	}

	var parts []string
	parts = append(parts, name, typ)

	if !getBool("nullable") {
		parts = append(parts, "NOT NULL")
	}
	if def := getName("default"); def != "" {
		parts = append(parts, "DEFAULT", def)
	}
	if getBool("unique") {
		parts = append(parts, "UNIQUE")
	}
	if getBool("primaryKey") {
		parts = append(parts, "PRIMARY KEY")
	}

	return strings.Join(parts, " "), nil
}

// squirrelPlaceholder is the PostgreSQL placeholder format for squirrel.
var squirrelPlaceholder = squirrel.Dollar
