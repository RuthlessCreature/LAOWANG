-- MySQL schema for LAOWANG (A 股分析系统)
-- Generated from a_stock_analyzer/db.py (tables + indexes + view) and frozen as SQL for ops.
--
-- Usage (example):
--   mysql -u astock -p -h 127.0.0.1 -P 3306 < sql/schema_mysql.sql
--
-- Notes:
-- - This file creates the database `astock` by default; change it if your config.ini uses another database.
-- - All tables use utf8mb4 + InnoDB.

SET NAMES utf8mb4;

CREATE DATABASE IF NOT EXISTS `astock`
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_0900_ai_ci;

USE `astock`;

-- ---------------------------------------------------------------------------
-- Core tables
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS `stock_info` (
  `stock_code` VARCHAR(16) NOT NULL,
  `name` VARCHAR(255) NULL,
  PRIMARY KEY (`stock_code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `stock_daily` (
  `stock_code` VARCHAR(16) NOT NULL,
  `date` VARCHAR(10) NOT NULL,
  `open` DOUBLE NULL,
  `high` DOUBLE NULL,
  `low` DOUBLE NULL,
  `close` DOUBLE NULL,
  `volume` DOUBLE NULL,
  `amount` DOUBLE NULL,
  PRIMARY KEY (`stock_code`, `date`),
  KEY `idx_stock_daily_date` (`date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `stock_indicators` (
  `stock_code` VARCHAR(16) NOT NULL,
  `date` VARCHAR(10) NOT NULL,
  `ma20` DOUBLE NULL,
  `ma60` DOUBLE NULL,
  `ma120` DOUBLE NULL,
  `rsi14` DOUBLE NULL,
  `macd_diff` DOUBLE NULL,
  `macd_dea` DOUBLE NULL,
  `macd_hist` DOUBLE NULL,
  `atr14` DOUBLE NULL,
  PRIMARY KEY (`stock_code`, `date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `stock_levels` (
  `stock_code` VARCHAR(16) NOT NULL,
  `calc_date` VARCHAR(10) NOT NULL,
  `support_level` DOUBLE NULL,
  `resistance_level` DOUBLE NULL,
  `support_type` VARCHAR(32) NULL,
  `resistance_type` VARCHAR(32) NULL,
  PRIMARY KEY (`stock_code`, `calc_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------------
-- Score tables
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS `stock_scores` (
  `stock_code` VARCHAR(16) NOT NULL,
  `score_date` VARCHAR(10) NOT NULL,
  `total_score` DOUBLE NULL,
  `trend_score` DOUBLE NULL,
  `pullback_score` DOUBLE NULL,
  `volume_price_score` DOUBLE NULL,
  `rsi_score` DOUBLE NULL,
  `macd_score` DOUBLE NULL,
  `market_cap_score` DOUBLE NULL,
  `tags` VARCHAR(255) NULL,
  PRIMARY KEY (`stock_code`, `score_date`),
  KEY `idx_stock_scores_date` (`score_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `stock_scores_v3` (
  `stock_code` VARCHAR(16) NOT NULL,
  `score_date` VARCHAR(10) NOT NULL,
  `total_score` DOUBLE NULL,
  `trend_score` DOUBLE NULL,
  `pullback_score` DOUBLE NULL,
  `volume_price_score` DOUBLE NULL,
  `rsi_score` DOUBLE NULL,
  `macd_score` DOUBLE NULL,
  `base_structure_score` DOUBLE NULL,
  `space_score` DOUBLE NULL,
  `market_cap_score` DOUBLE NULL,
  `status_tags` TEXT NULL,
  PRIMARY KEY (`stock_code`, `score_date`),
  KEY `idx_stock_scores_v3_date` (`score_date`),
  KEY `idx_stock_scores_v3_date_score` (`score_date`, `total_score`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------------
-- Backtest / performance table
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS `stock_future_perf` (
  `stock_code` VARCHAR(16) NOT NULL,
  `signal_date` VARCHAR(10) NOT NULL,
  `ne` INT NOT NULL,
  `max_price` DOUBLE NULL,
  `min_price` DOUBLE NULL,
  `final_price` DOUBLE NULL,
  `max_return` DOUBLE NULL,
  `min_return` DOUBLE NULL,
  `final_return` DOUBLE NULL,
  PRIMARY KEY (`stock_code`, `signal_date`, `ne`),
  KEY `idx_stock_future_perf_signal_date` (`signal_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ---------------------------------------------------------------------------  
-- Convenience view (MySQL only)
-- ---------------------------------------------------------------------------  

CREATE OR REPLACE VIEW `vw_stock_pool_v3_latest` AS
SELECT
  s.stock_code AS stock_code,
  d.close AS close,
  l.support_level AS support_level,
  l.resistance_level AS resistance_level,
  s.total_score AS total_score,
  s.status_tags AS status_tags,
  s.score_date AS score_date
FROM stock_scores_v3 s
LEFT JOIN stock_daily d
  ON d.stock_code = s.stock_code AND d.date = s.score_date
LEFT JOIN stock_levels l
  ON l.stock_code = s.stock_code AND l.calc_date = s.score_date
WHERE s.score_date = (SELECT MAX(score_date) FROM stock_scores_v3)
ORDER BY s.total_score DESC;

-- ---------------------------------------------------------------------------  
-- Materialized model outputs (for UI/BI)
-- ---------------------------------------------------------------------------  

CREATE TABLE IF NOT EXISTS `model_runs` (
  `model_name` VARCHAR(32) NOT NULL,
  `trade_date` VARCHAR(10) NOT NULL,
  `status` VARCHAR(16) NOT NULL,
  `row_count` INT NOT NULL,
  `message` TEXT NULL,
  `started_at` VARCHAR(19) NULL,
  `finished_at` VARCHAR(19) NULL,
  PRIMARY KEY (`model_name`, `trade_date`),
  KEY `idx_model_runs_trade_date` (`trade_date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `model_laowang_pool` (
  `trade_date` VARCHAR(10) NOT NULL,
  `stock_code` VARCHAR(16) NOT NULL,
  `stock_name` VARCHAR(255) NULL,
  `close` DOUBLE NULL,
  `support_level` DOUBLE NULL,
  `resistance_level` DOUBLE NULL,
  `total_score` DOUBLE NULL,
  `status_tags` TEXT NULL,
  `rank_no` INT NULL,
  PRIMARY KEY (`trade_date`, `stock_code`),
  KEY `idx_model_laowang_pool_trade_date` (`trade_date`),
  KEY `idx_model_laowang_pool_score` (`trade_date`, `total_score`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS `model_fhkq` (
  `trade_date` VARCHAR(10) NOT NULL,
  `stock_code` VARCHAR(16) NOT NULL,
  `stock_name` VARCHAR(255) NULL,
  `consecutive_limit_down` INT NULL,
  `last_limit_down` INT NULL,
  `volume_ratio` DOUBLE NULL,
  `amount_ratio` DOUBLE NULL,
  `open_board_flag` INT NULL,
  `liquidity_exhaust` INT NULL,
  `fhkq_score` INT NULL,
  `fhkq_level` VARCHAR(8) NULL,
  PRIMARY KEY (`trade_date`, `stock_code`),
  KEY `idx_model_fhkq_trade_date` (`trade_date`),
  KEY `idx_model_fhkq_score` (`trade_date`, `fhkq_score`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
