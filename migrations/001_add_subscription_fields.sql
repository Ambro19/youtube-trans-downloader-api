-- 001_add_subscription_fields.sql
-- Migration to add subscription and usage tracking fields to users table

-- Add subscription-related columns to users table
ALTER TABLE users ADD COLUMN subscription_tier VARCHAR(20) DEFAULT 'free';
ALTER TABLE users ADD COLUMN subscription_status VARCHAR(20) DEFAULT 'inactive';
ALTER TABLE users ADD COLUMN subscription_id VARCHAR(255);
ALTER TABLE users ADD COLUMN subscription_current_period_end DATETIME;
ALTER TABLE users ADD COLUMN stripe_customer_id VARCHAR(255);

-- Add usage tracking columns
ALTER TABLE users ADD COLUMN usage_clean_transcripts INTEGER DEFAULT 0;
ALTER TABLE users ADD COLUMN usage_unclean_transcripts INTEGER DEFAULT 0;
ALTER TABLE users ADD COLUMN usage_audio_downloads INTEGER DEFAULT 0;
ALTER TABLE users ADD COLUMN usage_video_downloads INTEGER DEFAULT 0;
ALTER TABLE users ADD COLUMN usage_reset_date DATETIME DEFAULT CURRENT_TIMESTAMP;

-- Create indexes for better performance
CREATE INDEX idx_users_subscription_tier ON users(subscription_tier);
CREATE INDEX idx_users_subscription_status ON users(subscription_status);
CREATE INDEX idx_users_stripe_customer_id ON users(stripe_customer_id);
CREATE INDEX idx_users_usage_reset_date ON users(usage_reset_date);

-- Update existing users to have default values
UPDATE users SET 
    subscription_tier = 'free',
    subscription_status = 'inactive',
    usage_clean_transcripts = 0,
    usage_unclean_transcripts = 0,
    usage_audio_downloads = 0,
    usage_video_downloads = 0,
    usage_reset_date = CURRENT_TIMESTAMP
WHERE subscription_tier IS NULL;