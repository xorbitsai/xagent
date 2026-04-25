# Uploaded File Path Migration Tool

## Background

The `storage_path` column in the `uploaded_files` table needs conversion between different formats:

- **Absolute paths**: `/uploads/user_1/web_task_123/output/file.txt`
- **Relative paths**: `web_task_123/output/file.txt` (without `user_{user_id}` prefix)

## Core Feature: Handle Multiple uploads_dir

**This tool is primarily designed for cases where `XAGENT_UPLOADS_DIR` has been changed multiple times.**

When the uploads directory location changes, the database contains paths with different prefixes:

```
/old/location/uploads/user_1/task_1/file.txt    ← Old uploads_dir
/new/location/uploads/user_1/task_2/file.txt    ← Middle uploads_dir
/current/uploads/user_1/task_3/file.txt         ← Current uploads_dir
```

Alembic automatic migration can only handle paths under the **current configured `UPLOADS_DIR`**. Other paths require manual source directory specification.

**Use `-d` parameter to specify old uploads directory paths**:
```bash
python scripts/migrate_uploads_file_abs_path.py migrate -d /old/location/uploads --confirm
```

**Multiple source directories can be specified** (chronological order from old to new):
```bash
python scripts/migrate_uploads_file_abs_path.py migrate \
  -d /oldest/uploads \
  -d /middle/uploads \
  -d /old/uploads \
  --confirm
```

**Example**: Suppose your uploads directory was migrated three times:

```
/var/www/xagent/uploads          ← Used in 2023
/home/user/xagent/uploads        ← Migrated here in 2024
/mnt/d/work/xagent/uploads        ← Migrated here in 2025
/current/xagent/uploads          ← Current config
```

Your database might contain paths like:

```
/var/www/xagent/uploads/user_1/task_1/file.txt
/home/user/xagent/uploads/user_1/task_2/file.txt
/mnt/d/work/xagent/uploads/user_1/task_3/file.txt
```

Run the command to unify them to relative paths:

```bash
python scripts/migrate_uploads_file_abs_path.py migrate \
  -d /var/www/xagent/uploads \
  -d /home/user/xagent/uploads \
  -d /mnt/d/work/xagent/uploads \
  --confirm
```

After conversion, all become:

```
task_1/file.txt
task_2/file.txt
task_3/file.txt
```

## When to Use This Tool

### Case 1: Alembic Migration Fails

When running `alembic upgrade` or `alembic downgrade`, if you see warnings like:

```
Warning: Could not convert /some/path to relative path (outside UPLOADS_DIR): ...
```

It means some file paths are not under the current `UPLOADS_DIR`, and automatic conversion failed. Use this tool to manually fix them.

### Case 2: Multiple Upload Directory Migrations

If `XAGENT_UPLOADS_DIR` configuration has been changed multiple times, the database may contain paths with different prefixes:

```
/old/uploads/user_1/task_1/file.txt
/new/uploads/user_1/task_2/file.txt
/current/uploads/user_1/task_3/file.txt
```

Automatic migration can only handle the current configured directory. Other paths need manual source directory specification.

### Case 3: Mixed Path Formats

The database contains both absolute and relative paths, and you need to unify the format.

## Usage

### 1. Check Current Status

```bash
python scripts/migrate_uploads_file_abs_path.py check
```

Example output:

```
Migration Status
┌────────────────────────────────────────────┬───────┬────────────┐
│ Category                                   │ Count │ Percentage │
├────────────────────────────────────────────┼───────┼────────────┤
│ Total records                              │    36 │            │
│ Absolute paths (need migration)            │    30 │    83.3%   │
│ Relative paths (already migrated)          │     6 │    16.7%   │
└────────────────────────────────────────────┴───────┴────────────┘

⚠ 30 absolute paths need migration.
```

### 2. Preview Migration (Dry-run Mode)

**Default is dry-run mode - no changes will be made to the database**. It only shows what will be converted.

```bash
# Specify the old uploads directory (without --confirm is preview)
python scripts/migrate_uploads_file_abs_path.py migrate -d /old/path/uploads

# Support multiple source directories
python scripts/migrate_uploads_file_abs_path.py migrate -d /old1/uploads -d /old2/uploads
```

Example output:

```
Migration Summary
┌──────────────────────────────┬───────┬───────┐
│ Category                     │ Count │ Status│
├──────────────────────────────┼───────┼───────┤
│ Total scanned                │    30 │       │
│ To migrate                   │    30 │   ✓   │
└──────────────────────────────┴───────┴───────┘

Preview (showing 10/30 records that will be migrated):
📄 c42346a0...
   Before: /old/uploads/user_1/web_task_25/output/chart.html
   After:  web_task_25/output/chart.html

ℹ️  DRY RUN MODE - No changes made to database
Use --confirm to actually migrate these records.
```

### 3. Execute Migration

**Add `--confirm` parameter to actually modify the database**.

```bash
# Execute migration after confirmation
python scripts/migrate_uploads_file_abs_path.py migrate -d /old/path/uploads --confirm
```

Example output:

```
Migration Summary
┌──────────────────────────────┬───────┬───────┐
│ Category                     │ Count │ Status│
├──────────────────────────────┼───────┼───────┤
│ To migrate                   │    30 │   ✓   │
└──────────────────────────────┴───────┴───────┘

✓ Successfully migrated 30 records
```

### 4. Advanced Options

```bash
# Show all records (default: first 10 only)
python scripts/migrate_uploads_file_abs_path.py migrate -d /old/path/uploads -v

# Custom batch size (default: 1000)
python scripts/migrate_uploads_file_abs_path.py migrate -d /old/path/uploads -b 500
```

## FAQ

### Q: How do I find the old uploads directory?

A: Check paths in the database:

```bash
sqlite3 xagent.db "SELECT DISTINCT substr(storage_path, 1, 50) FROM uploaded_files LIMIT 10;"
```

Or use the check command to see path samples.

### Q: Do I need to run Alembic after manual migration?

A: Depends on your target state:

- **Target is relative paths**: After manual migration, run `alembic upgrade head` to update version
- **Target is absolute paths**: After manual migration, run `alembic downgrade -1` to revert version

### Q: Can migration fail?

A: Paths not under the specified `--uploads-dir` will be skipped and counted as "Other absolute paths". Add more `-d` options to cover all paths.

### Q: How to rollback?

A: This script doesn't create backups. Backup your database first:

```bash
cp xagent.db xagent.db.backup
```

## Technical Details

### Path Format Rules

- **Relative paths**: Without `user_{user_id}` prefix, e.g., `web_task_123/output/file.txt`
- **Absolute paths**: Full path, e.g., `/uploads/user_1/web_task_123/output/file.txt`

### Cross-platform Support

The script automatically detects Windows-style paths (`C:\...`) and Unix-style paths, using the appropriate Path class for processing.

### Performance

- Uses `yield_per` for streaming to avoid memory overflow
- Batch commits (default: every 1000 records) for better performance
- Supports large databases (millions of records)
