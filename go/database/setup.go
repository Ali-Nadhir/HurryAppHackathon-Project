package database

import (
	sqlite_vec "github.com/asg017/sqlite-vec-go-bindings/cgo"
	"github.com/jmoiron/sqlx"
	_ "github.com/mattn/go-sqlite3"
)

var schema = `
CREATE TABLE IF NOT EXISTS person (
	id integer primary key autoincrement,
    name text not null,
	email text not null,
	phone text not null,
	address text not null,
	age integer not null,
	role text not null
);
CREATE TABLE IF NOT EXISTS fingerprint (
	id integer primary key autoincrement,
	person_id integer not null,
	finger integer not null
);


`

// create virtual table vec_fingerprint using vec0(
//   sample_embedding float[8]
// );

func Setup() *sqlx.DB {
	sqlite_vec.Auto()
	db := sqlx.MustConnect("sqlite3", "./data.db")

	db.MustExec(schema)

	return db
}
