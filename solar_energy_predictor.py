# solar_energy_predictor.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow loaded successfully")
except ImportError as e:
    print("⚠️ TensorFlow not available. Neural network will be skipped.")
    print(f"Details: {e}")
    TENSORFLOW_AVAILABLE = False

class SolarEnergyPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the solar energy dataset with robust error handling"""
        print("Loading and preprocessing data...")
        
        try:
            # Robust CSV reading with multiple fallback methods
            df = self._robust_csv_reader(file_path)
            
        except Exception as e:
            print(f"❌ Error loading data from {file_path}: {e}")
            print("🔄 Generating sample data instead...")
            df = self._create_sample_data()
        
        # Display basic info
        print(f"✅ Dataset shape: {df.shape}")
        print("\n📊 Columns:", df.columns.tolist())
        print("\n🔍 First few rows:")
        print(df.head())
        
        # Data cleaning
        df_clean = self._clean_data(df)
        
        # Feature engineering
        df_processed = self._engineer_features(df_clean)
        
        return df_processed
    
    def _robust_csv_reader(self, file_path):
        """Robust CSV reader that handles various formatting issues"""
        try:
            # Method 1: Try standard read with error handling
            df = pd.read_csv(file_path, encoding='utf-8')
            print("✅ CSV loaded with standard method")
            return df
            
        except pd.errors.ParserError as e:
            print(f"⚠️ Parser error: {e}")
            print("🔄 Trying alternative CSV reading method...")
            
            try:
                # Method 2: Skip bad lines with Python engine
                df = pd.read_csv(file_path, encoding='utf-8', 
                               on_bad_lines='skip', 
                               engine='python')
                print("✅ CSV loaded with error skipping method")
                return df
                
            except Exception as e2:
                print(f"⚠️ Alternative method failed: {e2}")
                print("🔄 Trying manual CSV parsing...")
                
                # Method 3: Manual parsing as last resort
                return self._manual_csv_parse(file_path)
    
    def _manual_csv_parse(self, file_path):
        """Manual CSV parsing for severely corrupted files"""
        print("📖 Manual CSV parsing initiated...")
        
        data = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Find header (first non-empty line)
        header = None
        for i, line in enumerate(lines):
            if line.strip():
                header = line.strip().split(',')
                break
        
        if header is None:
            raise ValueError("No header found in CSV file")
        
        print(f"📋 Detected header with {len(header)} columns: {header}")
        
        # Process data rows
        valid_rows = 0
        for i, line in enumerate(lines[1:], start=2):  # Skip header
            if line.strip():
                fields = line.strip().split(',')
                if len(fields) == len(header):
                    data.append(fields)
                    valid_rows += 1
                else:
                    print(f"⚠️ Skipping line {i}: expected {len(header)} fields, got {len(fields)}")
        
        print(f"✅ Successfully parsed {valid_rows} valid rows")
        return pd.DataFrame(data, columns=header)
    
    def _create_sample_data(self):
        """Create clean sample data if CSV file has issues"""
        print("🔧 Creating sample solar energy dataset...")
        dates = pd.date_range('2018-01-01', '2023-12-31', freq='D')
        
        # Simulate realistic seasonal patterns for Nairobi, Kenya
        n_days = len(dates)
        day_of_year = dates.dayofyear
        
        # Base solar radiation with seasonal pattern
        base_radiation = 4 + 2 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Add weather effects with realistic correlations
        temperature = 20 + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 2, n_days)
        humidity = 60 + 20 * np.sin(2 * np.pi * (day_of_year + 100) / 365) + np.random.normal(0, 10, n_days)
        wind_speed = 3 + np.random.exponential(1, n_days)
        precipitation = np.random.exponential(0.5, n_days)
        
        # Add realistic correlations and noise
        radiation = (base_radiation + 
                    0.1 * (temperature - 20) - 
                    0.02 * humidity + 
                    0.5 * wind_speed - 
                    2 * precipitation + 
                    np.random.normal(0, 0.5, n_days))
        
        # Ensure positive values and add some rounding for realism
        radiation = np.maximum(radiation, 0)
        
        df = pd.DataFrame({
            'YYYYMMDD': dates.strftime('%Y%m%d'),
            'ALLSKY_SFC_SW_DWN': np.round(radiation, 4),
            'T2M': np.round(temperature, 2),
            'RH2M': np.round(np.clip(humidity, 0, 100), 1),
            'WS2M': np.round(wind_speed, 2),
            'PRECTOTCORR': np.round(precipitation, 3),
            'YEAR': dates.year,
            'MO': dates.month,
            'DY': dates.day
        })
        
        # Save the clean sample data
        clean_filename = 'nasa_power_solar_data_clean.csv'
        df.to_csv(clean_filename, index=False)
        print(f"✅ Generated clean sample data: {df.shape}")
        print(f"💾 Saved as: {clean_filename}")
        
        return df
    
    def _clean_data(self, df):
        """Clean the raw dataset with robust error handling"""
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        print("🧹 Cleaning data...")
        
        # Convert date column with multiple fallback methods
        date_created = False
        if 'YYYYMMDD' in df_clean.columns:
            try:
                df_clean['DATE'] = pd.to_datetime(df_clean['YYYYMMDD'], format='%Y%m%d', errors='coerce')
                date_created = True
            except Exception as e:
                print(f"⚠️ Date conversion from YYYYMMDD failed: {e}")
        
        if not date_created and all(col in df_clean.columns for col in ['YEAR', 'MO', 'DY']):
            try:
                # Create date from year, month, day columns
                df_clean['DATE'] = pd.to_datetime(df_clean[['YEAR', 'MO', 'DY']])
                date_created = True
            except Exception as e:
                print(f"⚠️ Date conversion from YEAR/MO/DY failed: {e}")
        
        if not date_created:
            print("⚠️ Could not parse dates, creating sequential dates")
            df_clean['DATE'] = pd.date_range('2018-01-01', periods=len(df_clean), freq='D')
        
        # Set date as index and handle any remaining missing dates
        df_clean = df_clean.dropna(subset=['DATE'])
        df_clean.set_index('DATE', inplace=True)
        
        # Check for missing values
        print("\n📊 Missing values per column:")
        missing_info = df_clean.isnull().sum()
        print(missing_info)
        
        # Remove rows with missing target variable if it exists
        if 'ALLSKY_SFC_SW_DWN' in df_clean.columns:
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna(subset=['ALLSKY_SFC_SW_DWN'])
            removed_rows = initial_rows - len(df_clean)
            if removed_rows > 0:
                print(f"⚠️ Removed {removed_rows} rows with missing target values")
        
        # Fill remaining missing features with forward then backward fill
        df_clean = df_clean.ffill().bfill()
        
        # Drop any remaining rows with NaN values
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna()
        final_rows = len(df_clean)
        
        if initial_rows != final_rows:
            print(f"⚠️ Removed {initial_rows - final_rows} rows with persistent NaN values")
        
        print(f"✅ Data after cleaning: {df_clean.shape}")
        return df_clean
    
    def _engineer_features(self, df):
        """Create additional features for better prediction with error handling"""
        df_eng = df.copy()
        
        print("🔧 Engineering features...")
        
        # Time-based features
        df_eng['DAY_OF_YEAR'] = df_eng.index.dayofyear
        df_eng['MONTH'] = df_eng.index.month
        df_eng['SEASON'] = (df_eng.index.month % 12 + 3) // 3  # 1:Winter, 2:Spring, etc.
        
        # Cyclical encoding for seasonal patterns
        df_eng['DAY_SIN'] = np.sin(2 * np.pi * df_eng['DAY_OF_YEAR'] / 365)
        df_eng['DAY_COS'] = np.cos(2 * np.pi * df_eng['DAY_OF_YEAR'] / 365)
        
        # Weather interaction features (only create if base columns exist)
        if all(col in df_eng.columns for col in ['T2M', 'RH2M']):
            df_eng['TEMP_HUMIDITY'] = df_eng['T2M'] * df_eng['RH2M']
        
        if all(col in df_eng.columns for col in ['WS2M', 'T2M']):
            df_eng['WIND_TEMP'] = df_eng['WS2M'] * df_eng['T2M']
        
        # Lag features for temporal patterns (only if target exists)
        if 'ALLSKY_SFC_SW_DWN' in df_eng.columns:
            df_eng['SOLAR_LAG_1'] = df_eng['ALLSKY_SFC_SW_DWN'].shift(1)
            df_eng['SOLAR_LAG_7'] = df_eng['ALLSKY_SFC_SW_DWN'].shift(7)
            # Use min_periods to avoid losing too much data
            df_eng['SOLAR_7D_AVG'] = df_eng['ALLSKY_SFC_SW_DWN'].rolling(7, min_periods=1).mean()
        
        # Remove rows with NaN from lag features (but keep most of the data)
        initial_rows = len(df_eng)
        df_eng = df_eng.dropna()
        final_rows = len(df_eng)
        
        if initial_rows != final_rows:
            print(f"⚠️ Removed {initial_rows - final_rows} rows after feature engineering")
        
        print(f"✅ Data after feature engineering: {df_eng.shape}")
        return df_eng
    
    def prepare_features_target(self, df):
        """Prepare features and target variable with robust column handling"""
        # Target variable
        target = 'ALLSKY_SFC_SW_DWN'
        
        # Check if target exists
        if target not in df.columns:
            available_cols = df.columns.tolist()
            print(f"❌ Target column '{target}' not found!")
            print(f"📋 Available columns: {available_cols}")
            
            # Try to find alternative target column
            possible_targets = [col for col in available_cols if 'solar' in col.lower() or 'radiation' in col.lower()]
            if possible_targets:
                target = possible_targets[0]
                print(f"🔄 Using alternative target: {target}")
            else:
                raise ValueError(f"No suitable target column found. Available: {available_cols}")
        
        # Feature columns with fallbacks
        preferred_features = ['T2M', 'RH2M', 'WS2M', 'PRECTOTCORR', 
                             'DAY_SIN', 'DAY_COS', 'TEMP_HUMIDITY', 'WIND_TEMP',
                             'SOLAR_LAG_1', 'SOLAR_LAG_7', 'SOLAR_7D_AVG']
        
        # Only use columns that exist in dataframe
        feature_columns = [col for col in preferred_features if col in df.columns]
        
        # Ensure we have at least some features
        if not feature_columns:
            # Use basic features that should exist
            basic_features = ['DAY_SIN', 'DAY_COS']
            feature_columns = [col for col in basic_features if col in df.columns]
            if not feature_columns:
                raise ValueError("No feature columns available for training")
        
        X = df[feature_columns]
        y = df[target]
        
        print(f"✅ Features: {feature_columns}")
        print(f"📊 X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y, feature_columns
    
    def split_and_scale_data(self, X, y):
        """Split data and scale features with validation"""
        # Validate input shapes
        if len(X) != len(y):
            raise ValueError(f"X and y have different lengths: {len(X)} vs {len(y)}")
        
        if len(X) < 10:
            raise ValueError(f"Insufficient data for splitting: only {len(X)} samples")
        
        print("📊 Splitting and scaling data...")
        
        # Split data (use smaller test size if limited data)
        test_size = 0.2 if len(X) > 50 else 0.1
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=True  # Shuffle for better distribution
        )
        
        # Scale features - THIS WAS MISSING IN YOUR CODE!
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"🔍 Training set: {X_train_scaled.shape}")
        print(f"🔍 Testing set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple ML models with comprehensive error handling"""
        print("\n🤖 Training models...")
        
        # Traditional ML Models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        trained_count = 0
        for name, model in models.items():
            try:
                print(f"🔧 Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                self.models[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'metrics': self._calculate_metrics(y_test, y_pred)
                }
                print(f"✅ {name} trained - MAE: {self.models[name]['metrics']['mae']:.4f}")
                trained_count += 1
                
            except Exception as e:
                print(f"❌ Failed to train {name}: {e}")
                # Continue with other models
        
        # Neural Network (optional)
        if TENSORFLOW_AVAILABLE and trained_count > 0:
            try:
                print("🔧 Training Neural Network...")
                nn_model = self._build_neural_network(X_train.shape[1])
                history = nn_model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,  # Reduced for stability
                    batch_size=32,
                    verbose=0
                )
                
                y_pred_nn = nn_model.predict(X_test).flatten()
                
                self.models['Neural Network'] = {
                    'model': nn_model,
                    'predictions': y_pred_nn,
                    'metrics': self._calculate_metrics(y_test, y_pred_nn),
                    'history': history
                }
                
                print(f"✅ Neural Network trained - MAE: {self.models['Neural Network']['metrics']['mae']:.4f}")
                trained_count += 1
                
            except Exception as e:
                print(f"❌ Neural Network training failed: {e}")
        
        if trained_count == 0:
            raise Exception("❌ No models were successfully trained!")
        
        print(f"✅ Successfully trained {trained_count} models")
    
    def _build_neural_network(self, input_dim):
        """Build a neural network model"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics with validation"""
        # Handle cases where predictions might be problematic
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true({len(y_true)}) vs y_pred({len(y_pred)})")
        
        # Check for invalid values
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            print("⚠️ Warning: Invalid values in predictions")
            y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y_pred), posinf=np.nanmax(y_pred), neginf=np.nanmin(y_pred))
        
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
    
    def evaluate_models(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION")
        print("="*60)
        
        if not self.models:
            raise ValueError("❌ No models available for evaluation")
        
        results = []
        for name, model_info in self.models.items():
            metrics = model_info['metrics']
            results.append({
                'Model': name,
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'R²': metrics['r2']
            })
        
        results_df = pd.DataFrame(results)
        print("\n" + results_df.round(4).to_string(index=False))
        
        # Store results
        self.results['model_comparison'] = results_df
        
        # Identify best model based on MAE
        best_model_name = results_df.loc[results_df['MAE'].idxmin(), 'Model']
        best_r2 = results_df.loc[results_df['MAE'].idxmin(), 'R²']
        
        print(f"\n🏆 Best Model: {best_model_name} (MAE: {results_df['MAE'].min():.4f}, R²: {best_r2:.4f})")
        
        return best_model_name
    
    def plot_results(self, X_test, y_test, feature_names):
        """Create comprehensive visualization of results with error handling"""
        print("\n📈 Generating visualizations...")
        
        if not self.models:
            print("❌ No models available for plotting")
            return
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Solar Energy Prediction Analysis', fontsize=16, fontweight='bold')
            
            # 1. Actual vs Predicted for best model
            best_model_name = self.results['model_comparison'].loc[
                self.results['model_comparison']['MAE'].idxmin(), 'Model'
            ]
            best_predictions = self.models[best_model_name]['predictions']
            
            axes[0,0].scatter(y_test, best_predictions, alpha=0.6, color='blue')
            axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[0,0].set_xlabel('Actual Solar Radiation (kWh/m²/day)')
            axes[0,0].set_ylabel('Predicted Solar Radiation (kWh/m²/day)')
            axes[0,0].set_title(f'Actual vs Predicted - {best_model_name}')
            axes[0,0].grid(True, alpha=0.3)
            
            # 2. Model comparison
            models = self.results['model_comparison']['Model']
            mae_scores = self.results['model_comparison']['MAE']
            
            bars = axes[0,1].bar(models, mae_scores, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
            axes[0,1].set_ylabel('MAE (Lower is Better)')
            axes[0,1].set_title('Model Performance Comparison (MAE)')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, mae_scores):
                axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{value:.3f}', ha='center', va='bottom')
            
            # 3. Feature importance (if Random Forest exists)
            if 'Random Forest' in self.models:
                rf_model = self.models['Random Forest']['model']
                importances = rf_model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Only plot top features if many exist
                n_features = min(10, len(feature_names))
                axes[0,2].barh(range(n_features), importances[indices][:n_features], align='center')
                axes[0,2].set_yticks(range(n_features))
                axes[0,2].set_yticklabels([feature_names[i] for i in indices[:n_features]])
                axes[0,2].set_xlabel('Feature Importance')
                axes[0,2].set_title('Random Forest Feature Importance (Top 10)')
            else:
                axes[0,2].text(0.5, 0.5, 'Feature Importance\nNot Available', 
                              ha='center', va='center', transform=axes[0,2].transAxes)
                axes[0,2].set_title('Feature Importance')
            
            # 4. Time series plot (sample of test data)
            sample_size = min(100, len(y_test))
            if sample_size > 0:
                test_dates = range(sample_size)
                axes[1,0].plot(test_dates, y_test.values[:sample_size], label='Actual', linewidth=2, color='blue')
                axes[1,0].plot(test_dates, best_predictions[:sample_size], label='Predicted', alpha=0.8, color='red')
                axes[1,0].set_xlabel('Time Index')
                axes[1,0].set_ylabel('Solar Radiation (kWh/m²/day)')
                axes[1,0].set_title('Time Series: Actual vs Predicted')
                axes[1,0].legend()
                axes[1,0].grid(True, alpha=0.3)
            else:
                axes[1,0].text(0.5, 0.5, 'Insufficient data\nfor time series', 
                              ha='center', va='center', transform=axes[1,0].transAxes)
                axes[1,0].set_title('Time Series Plot')
            
            # 5. Residual plot
            residuals = y_test - best_predictions
            axes[1,1].scatter(best_predictions, residuals, alpha=0.6, color='green')
            axes[1,1].axhline(y=0, color='red', linestyle='--', linewidth=2)
            axes[1,1].set_xlabel('Predicted Values')
            axes[1,1].set_ylabel('Residuals')
            axes[1,1].set_title('Residual Analysis')
            axes[1,1].grid(True, alpha=0.3)
            
            # 6. Error distribution
            axes[1,2].hist(residuals, bins=30, alpha=0.7, edgecolor='black', color='orange')
            axes[1,2].axvline(x=0, color='red', linestyle='--', linewidth=2)
            axes[1,2].set_xlabel('Prediction Error')
            axes[1,2].set_ylabel('Frequency')
            axes[1,2].set_title('Distribution of Prediction Errors')
            axes[1,2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('solar_energy_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
            plt.show()
            print("✅ Visualizations saved as 'solar_energy_analysis.png'")
            
        except Exception as e:
            print(f"❌ Error generating plots: {e}")
            print("⚠️ Continuing without visualizations...")
    
    def ethical_analysis(self):
        """Analyze ethical considerations and biases"""
        print("\n" + "="*60)
        print("🌍 ETHICAL ANALYSIS & SUSTAINABILITY IMPACT")
        print("="*60)
        
        ethical_considerations = {
            "🔍 Data Bias Risks": [
                "Geographic bias: Model trained on Nairobi data may not generalize to other regions",
                "Temporal bias: Limited to 2018-2023, may not capture long-term climate patterns",
                "Weather station bias: Single location may not represent entire region",
                "Data quality: Relies on satellite estimates rather than ground measurements"
            ],
            "🛡️ Mitigation Strategies": [
                "Incorporate data from multiple geographic locations for transfer learning",
                "Use ensemble methods to reduce location-specific biases",
                "Implement uncertainty quantification in predictions",
                "Regular model retraining with new data and validation"
            ],
            "💚 Sustainability Impact": [
                "Enables better solar resource assessment for clean energy planning",
                "Supports grid stability through improved renewable energy forecasting",
                "Reduces reliance on fossil fuels by optimizing solar integration",
                "Promotes energy access in underserved communities through better planning"
            ],
            "🎯 SDG 7 Alignment": [
                "Affordable and Clean Energy: Optimizes solar power generation efficiency",
                "Climate Action: Supports transition to renewable energy sources",
                "Sustainable Cities: Enables smart grid operations and urban planning",
                "Industry Innovation: Advances predictive analytics for energy sector"
            ]
        }
        
        for category, points in ethical_considerations.items():
            print(f"\n{category}:")
            for point in points:
                print(f"  • {point}")

def main():
    """Main execution function with comprehensive error handling"""
    print("🚀 Starting SolarSense AI - Solar Energy Prediction")
    print("=" * 60)
    
    try:
        # Initialize the predictor
        predictor = SolarEnergyPredictor()
        
        # Try to load and preprocess data
        data_file = 'nasa_power_solar_data.csv'
        df_processed = predictor.load_and_preprocess_data(data_file)
        
        # Prepare features and target
        X, y, feature_names = predictor.prepare_features_target(df_processed)
        
        # Split and scale data
        X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test = predictor.split_and_scale_data(X, y)
        
        # Train models
        predictor.train_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Evaluate models
        best_model = predictor.evaluate_models(X_test_scaled, y_test)
        
        # Plot results
        predictor.plot_results(X_test_scaled, y_test, feature_names)
        
        # Ethical analysis
        predictor.ethical_analysis()
        
        # Print final summary
        print("\n" + "=" * 60)
        print("🎉 PROJECT SUMMARY")
        print("=" * 60)
        print("🌍 SDG Problem: SDG 7 - Affordable and Clean Energy")
        print("🤖 ML Approach: Multi-model regression for solar energy prediction")
        print(f"🏆 Best Model: {best_model}")
        
        best_metrics = predictor.models[best_model]['metrics']
        print(f"📊 Best Model Performance:")
        print(f"  • MAE: {best_metrics['mae']:.4f} kWh/m²/day")
        print(f"  • RMSE: {best_metrics['rmse']:.4f} kWh/m²/day")
        print(f"  • R²: {best_metrics['r2']:.4f}")
        
        print("\n💡 Key Impact: Enables better solar planning and grid integration")
        print("🌱 Sustainability: Supports transition to renewable energy sources")
        print("✅ Project completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Fatal error in main execution: {e}")
        print("💡 Troubleshooting tips:")
        print("  - Check if data file exists and is accessible")
        print("  - Verify all required packages are installed")
        print("  - Ensure sufficient memory and disk space")
        print("  - Try running with smaller dataset if memory issues occur")
        raise

if __name__ == "__main__":
    main()