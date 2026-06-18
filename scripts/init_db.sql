CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    quality VARCHAR(10) NOT NULL CHECK (quality IN ('good', 'bad')),
    confidence FLOAT NOT NULL,
    processing_time_ms FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(45)
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);

CREATE INDEX IF NOT EXISTS idx_predictions_quality ON predictions(quality);

\echo 'Database initialized successfully!'