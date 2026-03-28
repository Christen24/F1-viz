-- =============================================================================
-- Formula Chat — Database User Setup
-- Run once as a superuser after schema.sql
--
-- IMPORTANT: Replace 'changeme_in_production' with strong passwords before
-- running this script. Update DATABASE_URL in your .env files to match.
-- =============================================================================

-- Read-only user for the API
CREATE USER f1_readonly WITH PASSWORD 'changeme_in_production';

GRANT CONNECT ON DATABASE f1 TO f1_readonly;
GRANT USAGE ON SCHEMA public TO f1_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO f1_readonly;

-- Ensure future tables created by f1_user are also readable
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO f1_readonly;

-- Admin user for import scripts (read + write)
CREATE USER f1_user WITH PASSWORD 'changeme_in_production';

GRANT CONNECT ON DATABASE f1 TO f1_user;
GRANT USAGE, CREATE ON SCHEMA public TO f1_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO f1_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO f1_user;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO f1_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE ON SEQUENCES TO f1_user;
