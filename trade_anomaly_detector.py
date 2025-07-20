"""Trade Anomaly Detection System - Python Component

Enterprise-grade anomaly detection for trading data and PnL analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import tensorflow as tf
from tensorflow import keras
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats as stats

# System imports
import os
import sys
import threading
import queue
import schedule
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trade_anomaly_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of anomalies detected."""
    UNUSUAL_VOLUME = "Unusual Trading Volume"
    EXTREME_PNL = "Extreme P&L Movement"
    PATTERN_DEVIATION = "Pattern Deviation"
    STATISTICAL_OUTLIER = "Statistical Outlier"
    CORRELATION_BREAK = "Correlation Breakdown"
    LATENCY_SPIKE = "Execution Latency Spike"
    CONCENTRATION_RISK = "Position Concentration Risk"
    MARKET_IMPACT = "Abnormal Market Impact"
    SLIPPAGE_ANOMALY = "Excessive Slippage"
    RISK_LIMIT_BREACH = "Risk Limit Breach"

class Severity(Enum):
    """Anomaly severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class TradeData:
    """Represents a single trade."""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    price: float
    commission: float
    trader_id: str
    strategy: str
    venue: str
    latency_ms: float
    slippage: float = 0.0
    market_impact: float = 0.0

@dataclass
class PnLData:
    """Represents P&L data."""
    timestamp: datetime
    trader_id: str
    strategy: str
    symbol: str
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    position_size: float
    vwap: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float

@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    anomaly_id: str
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: Severity
    confidence: float
    affected_entity: str  # trader_id, symbol, or strategy
    description: str
    metrics: Dict[str, float]
    recommended_action: str
    ml_model_used: str
    false_positive: bool = False

class MLAnomalyDetector:
    """Machine Learning based anomaly detection for trading data."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the anomaly detector."""
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler()
        self.models = self._initialize_models()
        self.feature_extractors = self._initialize_feature_extractors()
        self.db_connection = self._setup_database()
        self.detection_history = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file."""
        default_config = {
            "database": {
                "path": "trade_anomalies.db",
                "backup_path": "trade_anomalies_backup.db"
            },
            "models": {
                "isolation_forest": {
                    "contamination": 0.05,
                    "n_estimators": 200,
                    "max_samples": "auto"
                },
                "autoencoder": {
                    "encoding_dim": 32,
                    "epochs": 100,
                    "batch_size": 32
                },
                "dbscan": {
                    "eps": 0.5,
                    "min_samples": 5
                }
            },
            "thresholds": {
                "pnl_zscore": 3.0,
                "volume_multiplier": 2.5,
                "latency_percentile": 99,
                "correlation_break": 0.3
            },
            "scheduling": {
                "detection_interval_minutes": 5,
                "model_retrain_hours": 24,
                "report_generation_time": "16:00"
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        
        return default_config
    
    def _setup_database(self) -> sqlite3.Connection:
        """Setup SQLite database for storing results."""
        conn = sqlite3.connect(
            self.config["database"]["path"],
            check_same_thread=False
        )
        
        # Create tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                timestamp DATETIME,
                symbol TEXT,
                side TEXT,
                quantity REAL,
                price REAL,
                commission REAL,
                trader_id TEXT,
                strategy TEXT,
                venue TEXT,
                latency_ms REAL,
                slippage REAL,
                market_impact REAL
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pnl_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                trader_id TEXT,
                strategy TEXT,
                symbol TEXT,
                realized_pnl REAL,
                unrealized_pnl REAL,
                total_pnl REAL,
                position_size REAL,
                vwap REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS anomalies (
                anomaly_id TEXT PRIMARY KEY,
                timestamp DATETIME,
                anomaly_type TEXT,
                severity INTEGER,
                confidence REAL,
                affected_entity TEXT,
                description TEXT,
                metrics TEXT,
                recommended_action TEXT,
                ml_model_used TEXT,
                false_positive BOOLEAN DEFAULT 0
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                model_name TEXT,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                false_positive_rate REAL,
                true_positive_rate REAL
            )
        """)
        
        conn.commit()
        return conn
    
    def _initialize_models(self) -> Dict:
        """Initialize ML models for anomaly detection."""
        models = {}
        
        # Isolation Forest for general anomaly detection
        models['isolation_forest'] = IsolationForest(
            contamination=self.config["models"]["isolation_forest"]["contamination"],
            n_estimators=self.config["models"]["isolation_forest"]["n_estimators"],
            max_samples=self.config["models"]["isolation_forest"]["max_samples"],
            random_state=42
        )
        
        # Autoencoder for complex pattern detection
        models['autoencoder'] = self._build_autoencoder()
        
        # DBSCAN for clustering-based anomaly detection
        models['dbscan'] = DBSCAN(
            eps=self.config["models"]["dbscan"]["eps"],
            min_samples=self.config["models"]["dbscan"]["min_samples"]
        )
        
        # Statistical models
        models['zscore'] = None  # Will use scipy.stats
        models['mad'] = None  # Median Absolute Deviation
        
        return models
    
    def _build_autoencoder(self) -> keras.Model:
        """Build autoencoder neural network for anomaly detection."""
        encoding_dim = self.config["models"]["autoencoder"]["encoding_dim"]
        
        # Input layer
        input_layer = keras.layers.Input(shape=(50,))  # 50 features
        
        # Encoder
        encoded = keras.layers.Dense(128, activation='relu')(input_layer)
        encoded = keras.layers.Dropout(0.2)(encoded)
        encoded = keras.layers.Dense(64, activation='relu')(encoded)
        encoded = keras.layers.Dropout(0.2)(encoded)
        encoded = keras.layers.Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = keras.layers.Dense(64, activation='relu')(encoded)
        decoded = keras.layers.Dropout(0.2)(decoded)
        decoded = keras.layers.Dense(128, activation='relu')(decoded)
        decoded = keras.layers.Dropout(0.2)(decoded)
        decoded = keras.layers.Dense(50, activation='sigmoid')(decoded)
        
        # Autoencoder model
        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return autoencoder
    
    def _initialize_feature_extractors(self) -> Dict:
        """Initialize feature extraction functions."""
        return {
            'trade_features': self._extract_trade_features,
            'pnl_features': self._extract_pnl_features,
            'market_features': self._extract_market_features,
            'risk_features': self._extract_risk_features
        }
    
    def _extract_trade_features(self, trades: pd.DataFrame) -> np.ndarray:
        """Extract features from trade data."""
        features = []
        
        # Volume-based features
        features.append(trades.groupby('symbol')['quantity'].sum())
        features.append(trades.groupby('trader_id')['quantity'].sum())
        features.append(trades.groupby('strategy')['quantity'].sum())
        
        # Price-based features
        features.append(trades.groupby('symbol')['price'].mean())
        features.append(trades.groupby('symbol')['price'].std())
        
        # Latency features
        features.append(trades['latency_ms'].mean())
        features.append(trades['latency_ms'].std())
        features.append(trades['latency_ms'].quantile(0.99))
        
        # Slippage features
        features.append(trades['slippage'].mean())
        features.append(trades['slippage'].std())
        
        # Market impact features
        features.append(trades['market_impact'].mean())
        features.append(trades['market_impact'].max())
        
        # Time-based features
        trades['hour'] = pd.to_datetime(trades['timestamp']).dt.hour
        features.append(trades.groupby('hour')['quantity'].sum())
        
        # Convert to numpy array
        feature_array = np.concatenate([f.values.flatten() for f in features])
        
        # Pad or truncate to fixed size
        if len(feature_array) > 50:
            feature_array = feature_array[:50]
        elif len(feature_array) < 50:
            feature_array = np.pad(feature_array, (0, 50 - len(feature_array)))
        
        return feature_array.reshape(1, -1)
    
    def _extract_pnl_features(self, pnl_data: pd.DataFrame) -> np.ndarray:
        """Extract features from P&L data."""
        features = []
        
        # P&L statistics
        features.append(pnl_data['total_pnl'].mean())
        features.append(pnl_data['total_pnl'].std())
        features.append(pnl_data['total_pnl'].min())
        features.append(pnl_data['total_pnl'].max())
        
        # Risk metrics
        features.append(pnl_data['sharpe_ratio'].mean())
        features.append(pnl_data['max_drawdown'].mean())
        features.append(pnl_data['win_rate'].mean())
        
        # Position analysis
        features.append(pnl_data['position_size'].sum())
        features.append(pnl_data['position_size'].std())
        
        # Rolling statistics
        if len(pnl_data) > 20:
            features.append(pnl_data['total_pnl'].rolling(20).mean().iloc[-1])
            features.append(pnl_data['total_pnl'].rolling(20).std().iloc[-1])
        else:
            features.extend([0, 0])
        
        feature_array = np.array(features)
        
        # Pad to fixed size
        if len(feature_array) < 50:
            feature_array = np.pad(feature_array, (0, 50 - len(feature_array)))
        
        return feature_array.reshape(1, -1)
    
    def _extract_market_features(self, trades: pd.DataFrame) -> np.ndarray:
        """Extract market microstructure features."""
        features = []
        
        # Order flow imbalance
        buy_volume = trades[trades['side'] == 'BUY']['quantity'].sum()
        sell_volume = trades[trades['side'] == 'SELL']['quantity'].sum()
        order_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-6)
        features.append(order_imbalance)
        
        # Trade intensity
        time_range = (trades['timestamp'].max() - trades['timestamp'].min()).total_seconds() / 3600
        trade_intensity = len(trades) / (time_range + 1e-6)
        features.append(trade_intensity)
        
        # Price dispersion
        features.append(trades.groupby('symbol')['price'].std().mean())
        
        # Venue concentration
        venue_counts = trades['venue'].value_counts()
        venue_concentration = (venue_counts.iloc[0] / venue_counts.sum()) if len(venue_counts) > 0 else 0
        features.append(venue_concentration)
        
        return np.array(features)
    
    def _extract_risk_features(self, trades: pd.DataFrame, pnl_data: pd.DataFrame) -> np.ndarray:
        """Extract risk-related features."""
        features = []
        
        # Concentration risk
        symbol_exposure = trades.groupby('symbol')['quantity'].sum() * trades.groupby('symbol')['price'].mean()
        total_exposure = symbol_exposure.sum()
        max_concentration = symbol_exposure.max() / total_exposure if total_exposure > 0 else 0
        features.append(max_concentration)
        
        # Trader concentration
        trader_pnl = pnl_data.groupby('trader_id')['total_pnl'].sum()
        total_pnl = trader_pnl.sum()
        trader_concentration = trader_pnl.abs().max() / (trader_pnl.abs().sum() + 1e-6)
        features.append(trader_concentration)
        
        # Strategy risk
        strategy_drawdown = pnl_data.groupby('strategy')['max_drawdown'].max()
        features.append(strategy_drawdown.mean())
        
        return np.array(features)
    
    def detect_anomalies(self, trades: pd.DataFrame, pnl_data: pd.DataFrame) -> List[Anomaly]:
        """Main anomaly detection function."""
        anomalies = []
        
        try:
            # Extract features
            trade_features = self._extract_trade_features(trades)
            pnl_features = self._extract_pnl_features(pnl_data)
            
            # Combine features
            all_features = np.concatenate([trade_features, pnl_features], axis=1)
            
            # Scale features
            if hasattr(self.scaler, 'mean_'):
                scaled_features = self.scaler.transform(all_features)
            else:
                scaled_features = self.scaler.fit_transform(all_features)
            
            # Run different detection methods
            anomalies.extend(self._isolation_forest_detection(scaled_features, trades, pnl_data))
            anomalies.extend(self._statistical_detection(trades, pnl_data))
            anomalies.extend(self._pattern_detection(trades, pnl_data))
            anomalies.extend(self._risk_limit_detection(trades, pnl_data))
            
            # Neural network detection (if trained)
            if hasattr(self.models['autoencoder'], 'predict'):
                anomalies.extend(self._autoencoder_detection(scaled_features, trades, pnl_data))
            
            # Deduplicate and prioritize anomalies
            anomalies = self._deduplicate_anomalies(anomalies)
            
            # Store in database
            self._store_anomalies(anomalies)
            
            logger.info(f"Detected {len(anomalies)} anomalies")
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            
        return anomalies
    
    def _isolation_forest_detection(self, features: np.ndarray, 
                                   trades: pd.DataFrame, 
                                   pnl_data: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies using Isolation Forest."""
        anomalies = []
        
        try:
            # Fit or predict
            if not hasattr(self.models['isolation_forest'], 'offset_'):
                self.models['isolation_forest'].fit(features)
            
            predictions = self.models['isolation_forest'].predict(features)
            scores = self.models['isolation_forest'].score_samples(features)
            
            if predictions[0] == -1:  # Anomaly detected
                confidence = abs(scores[0])
                
                # Determine anomaly type based on feature importance
                anomaly_type = self._determine_anomaly_type(features, trades, pnl_data)
                
                anomaly = Anomaly(
                    anomaly_id=f"IF_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    timestamp=datetime.now(),
                    anomaly_type=anomaly_type,
                    severity=self._calculate_severity(confidence, anomaly_type),
                    confidence=min(confidence, 1.0),
                    affected_entity=self._identify_affected_entity(trades, pnl_data),
                    description=f"{anomaly_type.value} detected by Isolation Forest",
                    metrics={'anomaly_score': float(scores[0])},
                    recommended_action=self._get_recommended_action(anomaly_type),
                    ml_model_used="Isolation Forest"
                )
                anomalies.append(anomaly)
                
        except Exception as e:
            logger.error(f"Isolation Forest detection error: {str(e)}")
            
        return anomalies
    
    def _statistical_detection(self, trades: pd.DataFrame, 
                             pnl_data: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies using statistical methods."""
        anomalies = []
        
        # Z-score based detection for P&L
        pnl_zscore = np.abs(stats.zscore(pnl_data['total_pnl']))
        threshold = self.config["thresholds"]["pnl_zscore"]
        
        for idx, zscore in enumerate(pnl_zscore):
            if zscore > threshold:
                row = pnl_data.iloc[idx]
                
                anomaly = Anomaly(
                    anomaly_id=f"STAT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{idx}",
                    timestamp=row['timestamp'],
                    anomaly_type=AnomalyType.EXTREME_PNL,
                    severity=Severity.HIGH if zscore > threshold * 1.5 else Severity.MEDIUM,
                    confidence=min(zscore / (threshold * 2), 1.0),
                    affected_entity=row['trader_id'],
                    description=f"Extreme P&L detected: ${row['total_pnl']:,.2f} (Z-score: {zscore:.2f})",
                    metrics={
                        'pnl': row['total_pnl'],
                        'zscore': zscore,
                        'sharpe_ratio': row['sharpe_ratio']
                    },
                    recommended_action="Review trading strategy and risk limits",
                    ml_model_used="Statistical Z-Score"
                )
                anomalies.append(anomaly)
        
        # Volume spike detection
        volume_by_symbol = trades.groupby('symbol')['quantity'].sum()
        volume_mean = volume_by_symbol.mean()
        volume_std = volume_by_symbol.std()
        
        for symbol, volume in volume_by_symbol.items():
            if volume > volume_mean + (volume_std * self.config["thresholds"]["volume_multiplier"]):
                anomaly = Anomaly(
                    anomaly_id=f"VOL_{datetime.now().strftime('%Y%m%d%H%M%S')}_{symbol}",
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.UNUSUAL_VOLUME,
                    severity=Severity.MEDIUM,
                    confidence=0.8,
                    affected_entity=symbol,
                    description=f"Unusual volume detected for {symbol}: {volume:,.0f} shares",
                    metrics={
                        'volume': volume,
                        'volume_mean': volume_mean,
                        'volume_std': volume_std
                    },
                    recommended_action="Check for news or market events",
                    ml_model_used="Statistical Volume Analysis"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _pattern_detection(self, trades: pd.DataFrame, 
                          pnl_data: pd.DataFrame) -> List[Anomaly]:
        """Detect pattern-based anomalies."""
        anomalies = []
        
        # Detect unusual trading patterns
        trades['hour'] = pd.to_datetime(trades['timestamp']).dt.hour
        hourly_volume = trades.groupby('hour')['quantity'].sum()
        
        # Check for unusual hour trading
        for hour, volume in hourly_volume.items():
            if hour < 6 or hour > 20:  # Outside normal hours
                if volume > hourly_volume.mean():
                    anomaly = Anomaly(
                        anomaly_id=f"PAT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hour}",
                        timestamp=datetime.now(),
                        anomaly_type=AnomalyType.PATTERN_DEVIATION,
                        severity=Severity.LOW,
                        confidence=0.7,
                        affected_entity="Trading Hours",
                        description=f"Unusual trading activity at hour {hour}: {volume:,.0f} shares",
                        metrics={'hour': hour, 'volume': volume},
                        recommended_action="Verify after-hours trading authorization",
                        ml_model_used="Pattern Analysis"
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _risk_limit_detection(self, trades: pd.DataFrame, 
                            pnl_data: pd.DataFrame) -> List[Anomaly]:
        """Detect risk limit breaches."""
        anomalies = []
        
        # Position concentration check
        symbol_exposure = trades.groupby('symbol').agg({
            'quantity': 'sum',
            'price': 'mean'
        })
        symbol_exposure['exposure'] = symbol_exposure['quantity'] * symbol_exposure['price']
        total_exposure = symbol_exposure['exposure'].sum()
        
        for symbol, row in symbol_exposure.iterrows():
            concentration = row['exposure'] / total_exposure if total_exposure > 0 else 0
            
            if concentration > 0.25:  # 25% concentration threshold
                anomaly = Anomaly(
                    anomaly_id=f"RISK_{datetime.now().strftime('%Y%m%d%H%M%S')}_{symbol}",
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.CONCENTRATION_RISK,
                    severity=Severity.HIGH,
                    confidence=0.9,
                    affected_entity=symbol,
                    description=f"High position concentration in {symbol}: {concentration:.1%}",
                    metrics={
                        'concentration': concentration,
                        'exposure': row['exposure'],
                        'total_exposure': total_exposure
                    },
                    recommended_action="Consider position reduction for risk management",
                    ml_model_used="Risk Limit Monitor"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _autoencoder_detection(self, features: np.ndarray,
                             trades: pd.DataFrame,
                             pnl_data: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies using autoencoder."""
        anomalies = []
        
        try:
            # Get reconstruction
            reconstruction = self.models['autoencoder'].predict(features)
            
            # Calculate reconstruction error
            mse = np.mean(np.square(features - reconstruction), axis=1)
            threshold = np.percentile(mse, 95)  # Top 5% as anomalies
            
            if mse[0] > threshold:
                anomaly = Anomaly(
                    anomaly_id=f"AE_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.PATTERN_DEVIATION,
                    severity=Severity.MEDIUM,
                    confidence=min(mse[0] / (threshold * 2), 1.0),
                    affected_entity="Multiple",
                    description="Complex pattern anomaly detected by neural network",
                    metrics={'reconstruction_error': float(mse[0])},
                    recommended_action="Deep dive analysis required",
                    ml_model_used="Autoencoder Neural Network"
                )
                anomalies.append(anomaly)
                
        except Exception as e:
            logger.error(f"Autoencoder detection error: {str(e)}")
            
        return anomalies
    
    def _determine_anomaly_type(self, features: np.ndarray,
                              trades: pd.DataFrame,
                              pnl_data: pd.DataFrame) -> AnomalyType:
        """Determine the type of anomaly based on feature analysis."""
        # Simple heuristic - in production, use feature importance
        volume_variance = trades['quantity'].var()
        pnl_variance = pnl_data['total_pnl'].var()
        
        if volume_variance > trades['quantity'].mean() * 2:
            return AnomalyType.UNUSUAL_VOLUME
        elif pnl_variance > pnl_data['total_pnl'].mean() * 2:
            return AnomalyType.EXTREME_PNL
        else:
            return AnomalyType.STATISTICAL_OUTLIER
    
    def _calculate_severity(self, confidence: float, 
                          anomaly_type: AnomalyType) -> Severity:
        """Calculate anomaly severity based on confidence and type."""
        if anomaly_type in [AnomalyType.RISK_LIMIT_BREACH, AnomalyType.EXTREME_PNL]:
            if confidence > 0.8:
                return Severity.CRITICAL
            elif confidence > 0.6:
                return Severity.HIGH
            else:
                return Severity.MEDIUM
        else:
            if confidence > 0.8:
                return Severity.HIGH
            elif confidence > 0.6:
                return Severity.MEDIUM
            else:
                return Severity.LOW
    
    def _identify_affected_entity(self, trades: pd.DataFrame,
                                pnl_data: pd.DataFrame) -> str:
        """Identify the most affected entity (trader, symbol, strategy)."""
        # Find entity with highest variance
        trader_var = pnl_data.groupby('trader_id')['total_pnl'].var()
        symbol_var = trades.groupby('symbol')['price'].var()
        
        if trader_var.max() > symbol_var.max():
            return trader_var.idxmax()
        else:
            return symbol_var.idxmax()
    
    def _get_recommended_action(self, anomaly_type: AnomalyType) -> str:
        """Get recommended action for anomaly type."""
        actions = {
            AnomalyType.UNUSUAL_VOLUME: "Review market conditions and trading algorithms",
            AnomalyType.EXTREME_PNL: "Investigate positions and market movements",
            AnomalyType.PATTERN_DEVIATION: "Check for system issues or strategy changes",
            AnomalyType.STATISTICAL_OUTLIER: "Perform detailed statistical analysis",
            AnomalyType.CORRELATION_BREAK: "Review portfolio hedges and correlations",
            AnomalyType.LATENCY_SPIKE: "Check network and system performance",
            AnomalyType.CONCENTRATION_RISK: "Rebalance portfolio to reduce concentration",
            AnomalyType.MARKET_IMPACT: "Review execution algorithms and timing",
            AnomalyType.SLIPPAGE_ANOMALY: "Analyze execution quality and liquidity",
            AnomalyType.RISK_LIMIT_BREACH: "Immediate position reduction required"
        }
        return actions.get(anomaly_type, "Manual review required")
    
    def _deduplicate_anomalies(self, anomalies: List[Anomaly]) -> List[Anomaly]:
        """Remove duplicate anomalies and prioritize by severity."""
        # Group by affected entity and anomaly type
        unique_anomalies = {}
        
        for anomaly in anomalies:
            key = (anomaly.affected_entity, anomaly.anomaly_type)
            
            if key not in unique_anomalies:
                unique_anomalies[key] = anomaly
            else:
                # Keep the one with higher severity/confidence
                if (anomaly.severity.value > unique_anomalies[key].severity.value or
                    anomaly.confidence > unique_anomalies[key].confidence):
                    unique_anomalies[key] = anomaly
        
        return list(unique_anomalies.values())
    
    def _store_anomalies(self, anomalies: List[Anomaly]):
        """Store anomalies in database."""
        for anomaly in anomalies:
            try:
                self.db_connection.execute("""
                    INSERT OR REPLACE INTO anomalies 
                    (anomaly_id, timestamp, anomaly_type, severity, confidence,
                     affected_entity, description, metrics, recommended_action,
                     ml_model_used, false_positive)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    anomaly.anomaly_id,
                    anomaly.timestamp,
                    anomaly.anomaly_type.value,
                    anomaly.severity.value,
                    anomaly.confidence,
                    anomaly.affected_entity,
                    anomaly.description,
                    json.dumps(anomaly.metrics),
                    anomaly.recommended_action,
                    anomaly.ml_model_used,
                    anomaly.false_positive
                ))
            except Exception as e:
                logger.error(f"Error storing anomaly {anomaly.anomaly_id}: {str(e)}")
        
        self.db_connection.commit()
    
    def train_models(self, historical_trades: pd.DataFrame,
                    historical_pnl: pd.DataFrame):
        """Train ML models on historical data."""
        logger.info("Training anomaly detection models...")
        
        try:
            # Extract features from historical data
            all_features = []
            
            # Process data in chunks
            chunk_size = 1000
            for i in range(0, len(historical_trades), chunk_size):
                trade_chunk = historical_trades.iloc[i:i+chunk_size]
                pnl_chunk = historical_pnl.iloc[i:i+chunk_size]
                
                trade_features = self._extract_trade_features(trade_chunk)
                pnl_features = self._extract_pnl_features(pnl_chunk)
                
                combined_features = np.concatenate([trade_features, pnl_features], axis=1)
                all_features.append(combined_features)
            
            # Combine all features
            X = np.vstack(all_features)
            
            # Fit scaler
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Isolation Forest
            self.models['isolation_forest'].fit(X_scaled)
            
            # Train Autoencoder
            self.models['autoencoder'].fit(
                X_scaled, X_scaled,
                epochs=self.config["models"]["autoencoder"]["epochs"],
                batch_size=self.config["models"]["autoencoder"]["batch_size"],
                validation_split=0.2,
                verbose=0
            )
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
    
    def evaluate_models(self, test_trades: pd.DataFrame,
                       test_pnl: pd.DataFrame,
                       known_anomalies: List[str]) -> Dict[str, float]:
        """Evaluate model performance."""
        # Implementation for model evaluation
        # Returns precision, recall, F1 score, etc.
        pass
    
    def mark_false_positive(self, anomaly_id: str):
        """Mark an anomaly as false positive for model improvement."""
        self.db_connection.execute("""
            UPDATE anomalies 
            SET false_positive = 1 
            WHERE anomaly_id = ?
        """, (anomaly_id,))
        self.db_connection.commit()
        
        logger.info(f"Marked anomaly {anomaly_id} as false positive")


# Data simulator for testing
class TradingDataSimulator:
    """Simulates realistic trading data with anomalies."""
    
    @staticmethod
    def generate_trades(n_trades: int = 1000, 
                       anomaly_rate: float = 0.05) -> pd.DataFrame:
        """Generate synthetic trade data."""
        trades = []
        
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'JPM', 'BAC', 'GS']
        traders = [f'TRADER_{i}' for i in range(1, 11)]
        strategies = ['MOMENTUM', 'MEAN_REVERSION', 'ARBITRAGE', 'MARKET_MAKING']
        venues = ['NYSE', 'NASDAQ', 'ARCA', 'BATS', 'IEX']
        
        base_time = datetime.now() - timedelta(hours=8)
        
        for i in range(n_trades):
            # Normal trade
            trade = TradeData(
                trade_id=f"T{i:06d}",
                timestamp=base_time + timedelta(seconds=i*30),
                symbol=np.random.choice(symbols),
                side=np.random.choice(['BUY', 'SELL']),
                quantity=np.random.randint(100, 10000),
                price=np.random.uniform(100, 500),
                commission=np.random.uniform(0.01, 0.05),
                trader_id=np.random.choice(traders),
                strategy=np.random.choice(strategies),
                venue=np.random.choice(venues),
                latency_ms=np.random.exponential(10),
                slippage=np.random.normal(0, 0.001),
                market_impact=np.random.uniform(0, 0.002)
            )
            
            # Inject anomaly
            if np.random.random() < anomaly_rate:
                anomaly_type = np.random.choice(['volume', 'latency', 'price'])
                
                if anomaly_type == 'volume':
                    trade.quantity *= 10  # Volume spike
                elif anomaly_type == 'latency':
                    trade.latency_ms = np.random.uniform(500, 2000)  # Latency spike
                elif anomaly_type == 'price':
                    trade.price *= np.random.choice([0.5, 2.0])  # Price anomaly
            
            trades.append(vars(trade))
        
        return pd.DataFrame(trades)
    
    @staticmethod
    def generate_pnl(n_records: int = 500,
                    anomaly_rate: float = 0.05) -> pd.DataFrame:
        """Generate synthetic P&L data."""
        pnl_records = []
        
        traders = [f'TRADER_{i}' for i in range(1, 11)]
        strategies = ['MOMENTUM', 'MEAN_REVERSION', 'ARBITRAGE', 'MARKET_MAKING']
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        
        base_time = datetime.now() - timedelta(hours=8)
        
        for i in range(n_records):
            # Normal P&L
            base_pnl = np.random.normal(1000, 5000)
            
            record = PnLData(
                timestamp=base_time + timedelta(minutes=i*5),
                trader_id=np.random.choice(traders),
                strategy=np.random.choice(strategies),
                symbol=np.random.choice(symbols),
                realized_pnl=base_pnl * 0.7,
                unrealized_pnl=base_pnl * 0.3,
                total_pnl=base_pnl,
                position_size=np.random.randint(1000, 50000),
                vwap=np.random.uniform(100, 500),
                sharpe_ratio=np.random.normal(1.5, 0.5),
                max_drawdown=np.random.uniform(0.02, 0.10),
                win_rate=np.random.uniform(0.45, 0.65)
            )
            
            # Inject anomaly
            if np.random.random() < anomaly_rate:
                anomaly_type = np.random.choice(['extreme_pnl', 'drawdown', 'sharpe'])
                
                if anomaly_type == 'extreme_pnl':
                    multiplier = np.random.choice([-10, 10])
                    record.total_pnl *= multiplier
                    record.realized_pnl *= multiplier
                elif anomaly_type == 'drawdown':
                    record.max_drawdown = np.random.uniform(0.20, 0.50)
                elif anomaly_type == 'sharpe':
                    record.sharpe_ratio = np.random.choice([-2, 5])
            
            pnl_records.append(vars(record))
        
        return pd.DataFrame(pnl_records)