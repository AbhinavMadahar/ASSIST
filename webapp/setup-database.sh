#!/bin/sh

set -ex

cockroach sql -e 'DROP DATABASE IF EXISTS example_flask_sqlalchemy'  --insecure
cockroach sql -e 'CREATE DATABASE example_flask_sqlalchemy' --insecure
cockroach sql -e 'GRANT ALL ON DATABASE example_flask_sqlalchemy TO example' --insecure

python -c 'import app; app.db.create_all()'
