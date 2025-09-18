# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, balanced_accuracy_score
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.base import clone
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import os
import sys
import pympi # Keep pympi import even if not directly used here, might be needed by elan_parser2
import re
import tempfile
import plotly.graph_objects as go
import plotly.express as px
import sklearn
from packaging import version
import traceback # Import traceback for exception handling

helper_script_dir = '/Data Analysis/New Analysis/helpers'

if helper_script_dir not in sys.path:
    sys.path.append(helper_script_dir)

# Attempt to import helper functions
try:
    from elan_parser2 import extract_features_from_elan
    from fixed_integration import create_features_dataframe_from_dict
except ImportError as e:
    st.error(f"Error importing helper scripts ('elan_parser2', 'fixed_integration'): {e}")
    st.error(f"Please ensure these files exist and the path is correct. Searched in: {helper_script_dir}")
    st.error(f"Current sys.path: {sys.path}")
    st.stop()


# Set page config
st.set_page_config(
    page_title="ELAN Aphasia Classifier",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .main { background-color: #0E1117; }
        .stButton>button { background-color: #4B6BFB; color: white; border-radius: 5px; border: none; padding: 10px 24px; transition: all 0.3s ease; }
        .stButton>button:hover { background-color: #6B8BFF; box-shadow: 0 5px 15px rgba(75, 107, 251, 0.4); }
        .css-1d391kg { padding: 2rem 1rem; } /* Adjust overall padding */
        .stDataFrame { background-color: #262730; padding: 1rem; border-radius: 5px; }
        h1 { color: #4B6BFB !important; font-weight: bold !important; }
        h2 { color: #6B8BFF !important; }
        h3 { color: #A1B8FF !important; }
        /* Consistent padding and margin for sections */
        .upload-section, .results-section, .prediction-section, .advanced-options, .elan-upload-section, .debug-section { background-color: #262730; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; }
        .advanced-options, .metric-card, .debug-section { background-color: #1E1E1E; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; }
        .metric-card { border-left: 4px solid #4B6BFB; }
        .elan-upload-section { border-left: 4px solid #4CAF50; }
        .debug-section { border-left: 4px solid #F7B500; background-color: #2C2C38; padding: 1.5rem; border-radius: 7px;}
        .stProgress > div > div { background-color: #4B6BFB; }
        .app-description { color: #FFFFFF; font-size: 1.1rem; margin-bottom: 1rem; } /* Reduced margin */
        .aphasic-probability { color: #FF6B4A; }
        .control-probability { color: #4CAF50; }
        .warning-box { background-color: #3A2E22; padding: 1rem; border-radius: 5px; border-left: 4px solid #F7B500; margin: 1rem 0; }
        .feature-importance-plot { background-color: #262730; padding: 1rem; border-radius: 5px; }
        /* Ensure plots have enough space */
        .stPlotlyChart, .stPyplot { margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="logo-title-container"><h1>🧠 ELAN Aphasia Classifier</h1></div>
    <p class="app-description">Train models to classify speech as Aphasic or Control based on linguistic features.</p>
""", unsafe_allow_html=True)


# --- Plotting Functions (with error handling and closing) ---

def plot_roc_curve(model, X_test, y_test, model_name):
    fig = None
    try:
        if hasattr(model, "predict_proba"): y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"): y_pred_proba = model.decision_function(X_test)
        else: st.warning(f"{model_name}: No predict_proba/decision_function for ROC."); return None
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba); roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})'); ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05]); ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_name}'); ax.legend(loc="lower right"); plt.tight_layout(); return fig
    except Exception as e: st.error(f"ROC plot error: {e}"); return None
    finally:
        if fig and plt.fignum_exists(fig.number): plt.close(fig)


def plot_learning_curve(model, X, y, model_name):
    """
    Failsafe learning curve implementation that always produces a valid output,
    even when facing SVM or LogisticRegression with class imbalance issues.
    """
    fig = None
    try:
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y

        # Immediately check for known problematic conditions
        unique_classes = np.unique(y_np)
        if len(unique_classes) < 2 or model_name in ['SVM', 'Logistic Regression']:
            # For SVM, LogisticRegression, or single-class data, don't attempt learning curve
            # Instead, create an informative plot
            fig, ax = plt.subplots(figsize=(10, 6))

            if len(unique_classes) < 2:
                message = "Cannot generate learning curve: Training data contains only one class"
            elif model_name in ['SVM', 'Logistic Regression']:
                message = f"Learning curve visualization not supported for {model_name} with this dataset"

            ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=14)

            # Add a helpful explanation
            if model_name == 'SVM':
                ax.text(0.5, 0.4, "SVM requires multiple samples of each class in every fold.",
                        ha='center', va='center', fontsize=12, color='#777')
                ax.text(0.5, 0.35, "Try using Random Forest for learning curve visualization.",
                        ha='center', va='center', fontsize=12, color='#777')
            elif model_name == 'Logistic Regression':
                ax.text(0.5, 0.4, "Logistic Regression requires multiple samples of each class in every fold.",
                        ha='center', va='center', fontsize=12, color='#777')
                ax.text(0.5, 0.35, "Try using Random Forest for learning curve visualization.",
                        ha='center', va='center', fontsize=12, color='#777')

            ax.set_xlim([0, 1]);
            ax.set_ylim([0, 1])
            ax.set_title(f'Learning Curve - {model_name}')
            ax.axis('off')
            plt.tight_layout()
            return fig

        # For RandomForest or other suitable models, proceed with normal learning curve
        n_samples = len(y_np)
        min_samples_for_cv = 5

        # Basic dataset size check
        if n_samples < min_samples_for_cv * 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Dataset too small ({n_samples} samples) for learning curve visualization",
                    ha='center', va='center', fontsize=14)
            ax.set_xlim([0, 1]);
            ax.set_ylim([0, 1])
            ax.set_title(f'Learning Curve - {model_name}')
            ax.axis('off')
            plt.tight_layout()
            return fig

        # Determine valid training sizes
        train_sizes_rel = np.linspace(0.1, 1.0, 10)
        valid_train_sizes = train_sizes_rel[(train_sizes_rel * n_samples * (1 - 1 / min_samples_for_cv)) >= 1]

        # If we have RandomForest or another suitable model, generate the learning curve
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=min_samples_for_cv, shuffle=True, random_state=42)

        train_sizes, train_scores, val_scores = learning_curve(
            model, X_np, y_np,
            train_sizes=valid_train_sizes,
            cv=cv,
            n_jobs=-1,
            scoring='balanced_accuracy',
            error_score='raise'
        )

        train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
        val_mean, val_std = np.mean(val_scores, axis=1), np.std(val_scores, axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_sizes, train_mean, 'o-', color='#4CAF50', label='Training score')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='#4CAF50')
        ax.plot(train_sizes, val_mean, 'o-', color='#FF6B4A', label='Cross-validation score')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='#FF6B4A')
        ax.set_xlabel('Training Examples')
        ax.set_ylabel('Balanced Accuracy Score')
        ax.set_title(f'Learning Curve - {model_name}')
        ax.legend(loc='best')
        ax.grid(True)
        plt.tight_layout()

        return fig

    except Exception as e:
        # Last resort fallback - create an error message plot
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            error_msg = str(e)
            if len(error_msg) > 100:
                error_msg = error_msg[:97] + "..."
            ax.text(0.5, 0.5, f"Error generating learning curve:\n{error_msg}",
                    ha='center', va='center', fontsize=12)
            ax.text(0.5, 0.4, "Try using Random Forest classifier instead for visualization.",
                    ha='center', va='center', fontsize=11, color='#777')
            ax.set_xlim([0, 1]);
            ax.set_ylim([0, 1])
            ax.set_title(f'Learning Curve - {model_name}')
            ax.axis('off')
            plt.tight_layout()
            return fig
        except:
            # If even creating an error plot fails, return None
            st.error(f"Learning curve completely failed with error: {e}")
            return None

    finally:
        if fig and plt.fignum_exists(fig.number):
            plt.close(fig)

def plot_calibration_curve(clf, X_train, y_train, X_test, y_test, n_bins=10):
    fig = None
    try:
        if hasattr(clf, "predict_proba"):
            prob_pos_train = clf.predict_proba(X_train)[:, 1]; prob_pos_test = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            decision_train = clf.decision_function(X_train); diff_train = decision_train.max() - decision_train.min()
            prob_pos_train = (decision_train - decision_train.min()) / diff_train if diff_train > 0 else np.full_like(decision_train, 0.5)
            decision_test = clf.decision_function(X_test); diff_test = decision_test.max() - decision_test.min()
            prob_pos_test = (decision_test - decision_test.min()) / diff_test if diff_test > 0 else np.full_like(decision_test, 0.5)
        else: st.warning("No predict_proba/decision_function for Calibration plot."); return None
        frac_pos_train, mean_pred_train = calibration_curve(y_train, prob_pos_train, n_bins=n_bins, strategy='uniform')
        frac_pos_test, mean_pred_test = calibration_curve(y_test, prob_pos_test, n_bins=n_bins, strategy='uniform')
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(mean_pred_train, frac_pos_train, "s-", label="Train Set"); ax.plot(mean_pred_test, frac_pos_test, "s-", label="Test Set")
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated"); ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration Curve"); ax.legend(loc="best"); plt.tight_layout(); return fig
    except ValueError as ve: st.warning(f"Calibration curve failed: {ve}. Both classes needed."); return None
    except Exception as e: st.error(f"Calibration plot error: {e}"); return None
    finally:
        if fig and plt.fignum_exists(fig.number): plt.close(fig)

def plot_feature_selection_importance(selected_features_mask, all_feature_names, importance_values=None):
    fig = None
    try:
        selected_names = [name for name, selected in zip(all_feature_names, selected_features_mask) if selected]
        if not selected_names: st.warning("No features selected to plot."); return None
        fig, ax = plt.subplots(figsize=(10, max(4, len(selected_names) * 0.4))) # Dynamic height
        if importance_values is not None and len(importance_values) == len(selected_names):
            df = pd.DataFrame({'Feature': selected_names, 'Importance': importance_values}).sort_values('Importance', ascending=True)
            sns.barplot(data=df, x='Importance', y='Feature', ax=ax, palette='viridis')
            ax.set_title('Selected Features and Their Importance')
        else:
            if importance_values is not None: st.warning("Importance/selection mismatch. Plotting selection only.")
            y_pos = range(len(selected_names)); ax.barh(y_pos, [1] * len(selected_names)); ax.set_yticks(y_pos); ax.set_yticklabels(selected_names)
            ax.set_xlabel('Selected'); ax.set_title('Selected Features'); ax.invert_yaxis()
        plt.tight_layout(); return fig
    except Exception as e: st.error(f"Plotting selection error: {e}"); return None
    finally:
        if fig and plt.fignum_exists(fig.number): plt.close(fig)

def plot_correlation_matrix(X):
    fig = None
    try:
        if not isinstance(X, pd.DataFrame) or X.empty: st.warning("No data for correlation matrix."); return None
        corr = X.corr(); fig, ax = plt.subplots(figsize=(10, 8)); mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0, square=True, linewidths=.5, annot=True, fmt=".2f", annot_kws={"size": 8}, ax=ax) # Smaller font
        plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0) # Improve label readability
        ax.set_title('Feature Correlation Matrix'); plt.tight_layout(); return fig
    except Exception as e: st.error(f"Correlation plot error: {e}"); return None
    finally:
        if fig and plt.fignum_exists(fig.number): plt.close(fig)

def plot_class_distribution(X, y, feature_names):
    fig = None
    try:
        if not isinstance(X, pd.DataFrame) or X.empty or not isinstance(y, (pd.Series, np.ndarray)) or len(X) != len(y) or not feature_names: st.warning("Invalid data for class distribution plot."); return None
        valid_features = [f for f in feature_names if f in X.columns]
        if not valid_features: st.warning("None of the specified features found in data."); return None
        combined = X[valid_features].copy(); combined['Class'] = ['Aphasic' if c else 'Control' for c in y]
        n_features = len(valid_features)
        fig, axs = plt.subplots(n_features, 1, figsize=(10, 4 * n_features), squeeze=False)
        for i, feature in enumerate(valid_features):
            ax = axs[i, 0]; sns.boxplot(x='Class', y=feature, data=combined, ax=ax, palette=['#4CAF50', '#FF6B4A']); ax.set_title(f'Distribution of {feature} by Class')
        plt.tight_layout(); return fig
    except Exception as e: st.error(f"Class distribution plot error: {e}"); return None
    finally:
        if fig and plt.fignum_exists(fig.number): plt.close(fig)

def visualize_feature_distributions(X_train, y_train, current_features_df, feature_names):
    figs = []
    try:
        if not isinstance(X_train, pd.DataFrame) or X_train.empty or not isinstance(y_train, (pd.Series, np.ndarray)) or len(X_train) != len(y_train) or not isinstance(current_features_df, pd.DataFrame) or current_features_df.empty: st.warning("Insufficient data for feature distribution plots."); return []
        valid_features = [f for f in feature_names if f in X_train.columns and f in current_features_df.columns]
        if not valid_features: st.warning("No common features found for distribution plots."); return []
        control_features = X_train.loc[~y_train, valid_features]; aphasic_features = X_train.loc[y_train, valid_features]
        for feature in valid_features:
             fig = go.Figure()
             if not control_features.empty and not control_features[feature].dropna().empty: fig.add_trace(go.Violin(y=control_features[feature].dropna(), name='Control (Train)', box_visible=True, meanline_visible=True, line_color='#4CAF50'))
             if not aphasic_features.empty and not aphasic_features[feature].dropna().empty: fig.add_trace(go.Violin(y=aphasic_features[feature].dropna(), name='Aphasic (Train)', box_visible=True, meanline_visible=True, line_color='#FF6B4A'))
             current_value = current_features_df[feature].values[0]
             if pd.notna(current_value): fig.add_trace(go.Scatter(x=['Current'], y=[current_value], mode='markers', marker=dict(size=15, color='#FFD700', symbol='star'), name='Current File'))
             fig.update_layout(title=f"{feature}", yaxis_title="Value", height=350, template="plotly_dark", showlegend=True, margin=dict(l=20, r=20, t=40, b=20)) # Adjust margins
             figs.append(fig)
        return figs
    except Exception as e: st.error(f"Visualize distributions error: {e}"); return []


# --- Data Loading Function --- (Modified from v2)
def load_and_prepare_data(stats_file_obj, pauses_file_obj):
    """Load and prepare data from uploaded file objects."""
    try:
        stats_df = pd.read_csv(io.BytesIO(stats_file_obj.getvalue()))
        pauses_df = pd.read_csv(io.BytesIO(pauses_file_obj.getvalue()))
    except Exception as e: st.error(f"Error reading CSVs: {e}"); return None, {}

    required_stats = ['participant_id', 'words_per_minute', 'total_pauses', 'recording_duration_minutes', 'grammatical_utterances', 'ungrammatical_utterances', 'filled_pauses', 'group']
    required_pauses = ['participant_id', 'duration']
    missing_stats = [c for c in required_stats if c not in stats_df.columns]
    missing_pauses = [c for c in required_pauses if c not in pauses_df.columns]
    if missing_stats: st.error(f"Stats CSV missing: {', '.join(missing_stats)}"); return None, {}
    if missing_pauses: st.error(f"Pauses CSV missing: {', '.join(missing_pauses)}"); return None, {}

    features_df = pd.DataFrame()
    wpm_scale_factor, ppm_scale_factor = 1.0, 1.0
    warnings_list = []

    wpm_mean = stats_df['words_per_minute'].mean()
    if pd.notna(wpm_mean) and wpm_mean < 5.0 and wpm_mean != 0: wpm_scale_factor = 60.0; warnings_list.append(f"Low WPM mean ({wpm_mean:.2f}). Applied x{wpm_scale_factor} factor.")
    duration = stats_df['recording_duration_minutes']
    ppm = stats_df.get('total_pauses_per_minute', stats_df['total_pauses'].divide(duration.where(duration > 0, np.nan)).fillna(0))
    ppm_mean = ppm.mean()
    if pd.notna(ppm_mean) and ppm_mean > 500.0: ppm_scale_factor = 0.01; warnings_list.append(f"High Pauses/min mean ({ppm_mean:.2f}). Applied x{ppm_scale_factor} factor.")

    processed_participants = []
    unique_ids = stats_df['participant_id'].unique()
    if len(unique_ids) == 0: st.error("No participant IDs found."); return None, {}

    for p_id in unique_ids:
        p_stats = stats_df[stats_df['participant_id'] == p_id].iloc[0]
        p_pauses = pauses_df[pauses_df['participant_id'] == p_id]
        total_p = p_stats.get('total_pauses', 0); filled_p = p_stats.get('filled_pauses', 0)
        filled_r = filled_p / total_p if pd.notna(total_p) and total_p > 0 else 0
        gram_u = p_stats.get('grammatical_utterances', 0); ungram_u = p_stats.get('ungrammatical_utterances', 0); gram_r = 0
        if pd.notna(gram_u) and pd.notna(ungram_u): total_u = gram_u + ungram_u; gram_r = gram_u / total_u if total_u > 0 else 0
        wpm = p_stats.get('words_per_minute', 0); wpm = wpm * wpm_scale_factor if pd.notna(wpm) else 0
        rec_dur = p_stats.get('recording_duration_minutes', 0)
        curr_ppm = p_stats.get('total_pauses_per_minute', (total_p / rec_dur if pd.notna(rec_dur) and rec_dur > 0 else 0) if pd.notna(total_p) else 0 )
        curr_ppm = curr_ppm * ppm_scale_factor if pd.notna(curr_ppm) else 0
        mean_pause = 0
        if not p_pauses.empty and 'duration' in p_pauses.columns: mean_pause = p_pauses['duration'].mean(); mean_pause = mean_pause if pd.notna(mean_pause) else 0
        features = {'participant_id': p_id, 'words_per_minute': wpm, 'total_pauses_per_minute': curr_ppm, 'grammaticality_ratio': gram_r, 'mean_pause_duration': mean_pause, 'filled_pause_ratio': filled_r, 'group': p_stats.get('group', 'unknown')}
        processed_participants.append(features)

    if not processed_participants: st.error("No participants processed."); return None, {}
    features_df = pd.DataFrame(processed_participants)

    imputed_cols = []
    for col in features_df.columns.drop(['participant_id', 'group']):
        if features_df[col].isnull().any(): mean_val = features_df[col].mean(); features_df[col].fillna(mean_val, inplace=True); imputed_cols.append(col)
    if imputed_cols: warnings_list.append(f"NaNs imputed with mean for: {', '.join(imputed_cols)}")
    for warning in warnings_list: st.warning(warning)
    scale_factors = {'words_per_minute': wpm_scale_factor, 'total_pauses_per_minute': ppm_scale_factor, 'mean_pause_duration': 1.0, 'grammaticality_ratio': 1.0, 'filled_pause_ratio': 1.0}

    # --- FIX: Auto-scale mean_pause_duration if likely in ms ---
    if 'mean_pause_duration' in features_df.columns:
        mean_mpd = features_df['mean_pause_duration'].mean()
        if mean_mpd > 10:
            features_df['mean_pause_duration'] = features_df['mean_pause_duration'] / 1000.0
            warnings_list.append("Converted mean_pause_duration from ms to seconds (auto-detected).")

    return features_df, scale_factors


# --- ELAN Feature Normalization --- (Modified from v2)
def normalize_feature_values(features_dict, training_data=None, scale_factors=None):
    """
    Normalize feature values from ELAN extraction with direct fix for scaling issues.
    """
    if not isinstance(features_dict, dict):
        return {}

    # Make a copy to avoid modifying the original
    normalized_dict = features_dict.copy()
    display_messages = []

    # DIRECT FIX for the specific scaling issues
    if 'words_per_minute' in normalized_dict:
        # Fix for WPM being multiplied by 100
        original_wpm = normalized_dict['words_per_minute']
        if original_wpm > 200:  # If WPM is unusually high, it's likely been multiplied
            normalized_dict['words_per_minute'] = original_wpm / 100.0
            display_messages.append(
                f"🔧 Fixed WPM scaling: {original_wpm:.2f} → {normalized_dict['words_per_minute']:.2f}")

    if 'total_pauses_per_minute' in normalized_dict:
        # Fix for pauses/min being divided by 100
        original_ppm = normalized_dict['total_pauses_per_minute']
        if original_ppm < 1.0 and original_ppm > 0:  # If pauses/min is unusually low, it's likely been divided
            normalized_dict['total_pauses_per_minute'] = original_ppm * 100.0
            display_messages.append(
                f"🔧 Fixed Pauses/min scaling: {original_ppm:.2f} → {normalized_dict['total_pauses_per_minute']:.2f}")

    # Additional normalization for other features
    expected_ranges = {
        'words_per_minute': (1, 300),
        'total_pauses_per_minute': (0, 200),
        'mean_pause_duration': (0.05, 10),
        'grammaticality_ratio': (0, 1),
        'filled_pause_ratio': (0, 1)
    }

    # Apply any remaining heuristics needed
    for feature, (expected_min, expected_max) in expected_ranges.items():
        if feature in normalized_dict:
            value = normalized_dict[feature]
            if pd.isna(value) or not isinstance(value, (int, float)):
                normalized_dict[feature] = 0
                display_messages.append(f"⚠️ Non-numeric '{feature}'. Set to 0.")
                continue

            # Apply additional heuristics only if needed after the direct fixes
            if expected_min <= value <= expected_max:
                continue

            original_value = value
            heuristic_applied = False

            if feature == 'mean_pause_duration' and value > 10:
                normalized_dict[feature] = value / 1000.0
                heuristic_applied = True
            elif feature in ['grammaticality_ratio', 'filled_pause_ratio'] and value > 1:
                normalized_dict[feature] = min(value / 100.0, 1.0)
                heuristic_applied = True

            if heuristic_applied:
                display_messages.append(
                    f"⚠️ Applied additional scaling to '{feature}': {original_value:.2f} → {normalized_dict[feature]:.2f}")
            else:
                display_messages.append(
                    f"⚠️ '{feature}' ({value:.2f}) outside expected range ({expected_min:.2f}-{expected_max:.2f}).")

    if display_messages:
        st.info("Normalization Notes:\n" + "\n".join(display_messages))

    return normalized_dict

# --- Initialize session state ---
def init_session_state():
    defaults = {
        'model': None, 'scaler': None, 'feature_names': None, 'model_type': None,
        'selected_features_mask': None, 'selected_feature_names': None,
        'X_original': None, 'y_original': None, 'participant_ids': None,
        'calibrated_model': None, 'debug_mode': False,
        'feature_scaling_factors': {'words_per_minute': 1.0, 'total_pauses_per_minute': 1.0, 'mean_pause_duration': 1.0, 'grammaticality_ratio': 1.0, 'filled_pause_ratio': 1.0},
        'data_loaded': False, 'model_trained': False,
        'X_test': None, 'y_test': None,
        'processed_stats_file_id': None, 'processed_pauses_file_id': None
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value

init_session_state()


# --- Define Tabs ---
tabs = st.tabs(["Train Model", "Predict from ELAN File", "Documentation"])


# ===========================
#       Train Model Tab
# ===========================
with tabs[0]:
    # --- File Upload Section ---
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.header("📤 Upload Training Data Files")
        col1, col2 = st.columns(2)
        with col1: stats_file_up = st.file_uploader("Upload Combined Statistics CSV", type=['csv'], key="stats_up", help="CSV with participant_id, group (aphasic/control), WPM, pause counts, duration, utterance counts, etc.")
        with col2: pauses_file_up = st.file_uploader("Upload Combined Pause Analysis CSV", type=['csv'], key="pauses_up", help="CSV with participant_id and individual pause durations (in seconds).")
        st.session_state.debug_mode = st.checkbox("Enable Detailed Mode", value=st.session_state.debug_mode, key="debug_mode_toggle", help="Show detailed plots and statistics during training.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Data Loading Trigger ---
    process_files_now = False
    if stats_file_up and pauses_file_up:
        current_stats_id = stats_file_up.file_id; current_pauses_id = pauses_file_up.file_id
        if (current_stats_id != st.session_state.get('processed_stats_file_id') or current_pauses_id != st.session_state.get('processed_pauses_file_id')):
            process_files_now = True

    # --- Data Loading and Processing Block ---
    if process_files_now:
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        with st.spinner("Loading and Preparing Data..."):
            st.session_state.data_loaded = False; st.session_state.model_trained = False
            features_df, scale_factors = load_and_prepare_data(stats_file_up, pauses_file_up)
            if features_df is not None and not features_df.empty:
                feature_names = ['words_per_minute', 'total_pauses_per_minute', 'grammaticality_ratio', 'mean_pause_duration', 'filled_pause_ratio']
                missing_features = [f for f in feature_names if f not in features_df.columns]
                if missing_features: st.error(f"Data missing features: {', '.join(missing_features)}"); st.session_state.processed_stats_file_id = None; st.session_state.processed_pauses_file_id = None
                else:
                     try:
                         X = features_df[feature_names].copy();
                         if 'group' not in features_df.columns: raise ValueError("'group' missing")
                         y = (features_df['group'].astype(str).str.lower() == 'aphasic');
                         if len(y.unique()) < 2: st.warning("Training data contains only one class.")
                         st.session_state.X_original = X; st.session_state.y_original = y; st.session_state.participant_ids = features_df.get('participant_id')
                         st.session_state.feature_names = feature_names; st.session_state.feature_scaling_factors = scale_factors; st.session_state.processed_stats_file_id = current_stats_id; st.session_state.processed_pauses_file_id = current_pauses_id
                         # Reset model state fully
                         st.session_state.model = None; st.session_state.scaler = None; st.session_state.calibrated_model = None; st.session_state.X_test = None; st.session_state.y_test = None; st.session_state.selected_features_mask = None; st.session_state.selected_feature_names = None; st.session_state.model_trained = False
                         st.session_state.data_loaded = True # Set loaded flag ONLY on success
                         st.success("Data loaded successfully! Configure training below.")
                     except Exception as e: st.error(f"Error processing data: {e}"); st.session_state.processed_stats_file_id = None; st.session_state.processed_pauses_file_id = None
            else: st.error("Data loading failed."); st.session_state.processed_stats_file_id = None; st.session_state.processed_pauses_file_id = None
        st.markdown('</div>', unsafe_allow_html=True)


    # --- Training, Evaluation, and Prediction Section ---
    if st.session_state.get('data_loaded', False):

        # Data Overview
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.header("📊 Data Overview")
        X_display, y_display = st.session_state.X_original, st.session_state.y_original
        col_sum1, col_sum2 = st.columns(2)
        with col_sum1: st.metric("Samples", len(X_display)); st.metric("Aphasic", int(sum(y_display))); st.metric("Control", int(sum(~y_display)))
        with col_sum2: st.dataframe(X_display.describe().T.style.format("{:.2f}"))
        st.markdown('</div>', unsafe_allow_html=True)

        # Debug Diagnostics
        if st.session_state.debug_mode:
            st.markdown('<div class="debug-section">', unsafe_allow_html=True)
            st.subheader("🕵️ Debug Diagnostics")
            col_d1, col_d2 = st.columns(2)
            with col_d1: # Pie Chart
                 st.write("**Class Distribution:**")
                 if sum(y_display) + sum(~y_display) > 0:
                      fig_bal = plt.figure(figsize=(5, 3)); plt.pie([sum(~y_display), sum(y_display)], labels=['Control', 'Aphasic'], autopct='%1.1f%%', colors=['#4CAF50', '#FF6B4A']); plt.title('Class Distribution'); st.pyplot(fig_bal); plt.close(fig_bal)
                 else: st.warning("No samples to plot distribution.")
            with col_d2: # Correlation
                 st.write("**Feature Correlation:**")
                 fig_corr = plot_correlation_matrix(X_display);
                 if fig_corr: st.pyplot(fig_corr); # Already closed in function
            st.write("**Feature Distributions by Class:**") # Box Plots
            fig_dist = plot_class_distribution(X_display, y_display, st.session_state.feature_names)
            if fig_dist: st.pyplot(fig_dist); # Already closed in function
            st.markdown('</div>', unsafe_allow_html=True)

        # Model Training Configuration
        st.markdown('<div class="advanced-options">', unsafe_allow_html=True)
        st.header("⚙️ Model Training Configuration")
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1: model_type_select = st.selectbox("Model Type", ['Random Forest', 'SVM', 'Logistic Regression', 
                                                                       'Voting Ensemble'], key="model_select"); 
        feature_select_method = st.selectbox("Feature Selection", ['None', 'Model-based Selection (RF)', 'Recursive Feature Elimination (RFE)'], 
                                             key="feat_select")
        with col_opt2: # Hyperparameters
            model_params = {}
            if model_type_select == 'Random Forest': model_params['max_depth'] = st.slider("Max Depth", 2, 20, 5, key="rf_depth"); model_params['min_samples_split'] = st.slider("Min Samples Split", 2, 10, 4, key="rf_split"); model_params['n_estimators'] = st.slider("# Trees", 10, 200, 100, key="rf_trees")
            elif model_type_select == 'SVM': model_params['C'] = st.slider("Reg (C)", 0.1, 10.0, 1.0, 0.1, key="svm_c"); model_params['kernel'] = st.selectbox("Kernel", ['rbf', 'linear', 'poly'], key="svm_kernel"); model_params['gamma'] = 'scale'
            else: # Logistic Regression
                model_params['C'] = st.slider("Reg (C)", 0.1, 10.0, 1.0, 0.1, key="lr_c"); model_params['solver'] = st.selectbox("Solver", ['liblinear', 'lbfgs', 'saga'], key="lr_solver"); penalty_options = {'liblinear': ['l1', 'l2'], 'lbfgs': ['l2', 'none'], 'saga': ['l1', 'l2', 'elasticnet', 'none']}; model_params['penalty'] = st.selectbox("Penalty", penalty_options.get(model_params['solver'], ['l2']), key="lr_penalty");
                if model_params.get('penalty') == 'elasticnet': model_params['l1_ratio'] = st.slider("L1 Ratio", 0.0, 1.0, 0.5, 0.05, key="lr_l1ratio")
                if model_params.get('penalty') == 'none' and version.parse(sklearn.__version__) >= version.parse('1.0.0'): model_params['penalty'] = None
            if model_type_select == 'Voting Ensemble':
                st.write("**Ensemble Configuration**")
                use_rf = st.checkbox("Include Random Forest", True, key="use_rf")
                use_svm = st.checkbox("Include SVM", True, key="use_svm")
                use_lr = st.checkbox("Include Logistic Regression", True, key="use_lr")
                ensemble_voting = st.selectbox("Voting Type", ['soft', 'hard'], key="ensemble_voting",
                                               help= "'soft' uses predicted probabilities, 'hard' uses class predictions")
                if not (use_rf or use_svm or use_lr):
                    st.warning("Please select at least one model for the ensemle")
        with col_opt3: # CV & Other Options
            cv_options = ['K-Fold (5)'] + (['Leave-One-Subject-Out'] if st.session_state.participant_ids is not None else []); cv_method_select = st.selectbox("Cross-validation", cv_options, key="cv_method")
            use_calibration_select = st.checkbox("Calibrate Probabilities", True, key="use_calib", help="Uses CalibratedClassifierCV after training."); class_weights_select = st.selectbox("Class Weights", ['balanced', None], key="class_weight", help="Address class imbalance.")
        st.markdown('</div>', unsafe_allow_html=True)


        # --- Train Button ---
        if st.button("Train Model", key="train_button"):
            st.session_state.model_trained = False # Reset flag
            st.markdown('<div class="results-section">', unsafe_allow_html=True)
            st.header("⏳ Training Process")
            with st.spinner(f"Running Training & CV for {model_type_select}..."):
                # Get data from session state
                X_train_full, y_train_full = st.session_state.X_original, st.session_state.y_original
                feature_names_train = st.session_state.feature_names

                # Perform Train/Test Split
                try:
                    stratify_param = y_train_full if len(y_train_full.unique()) > 1 else None
                    X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=stratify_param)
                    st.session_state.X_test, st.session_state.y_test = X_test, y_test # Store test set
                except Exception as e: st.error(f"Train/Test Split Error: {e}"); st.stop()

                # Scaling
                scaler = StandardScaler(); X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index); X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index); st.session_state.scaler = scaler

                # Prepare data for this run (will be modified by feature selection)
                X_train_run, X_test_run = X_train_scaled.copy(), X_test_scaled.copy()
                current_feature_names_run = feature_names_train.copy(); selected_mask_run = np.ones(len(current_feature_names_run), dtype=bool); selected_names_run = current_feature_names_run.copy()

                # Build Base Model instance
                try:
                    if model_type_select == 'Random Forest': base_model = RandomForestClassifier(n_estimators=model_params['n_estimators'], max_depth=model_params['max_depth'], min_samples_split=model_params['min_samples_split'], class_weight=class_weights_select, random_state=42, n_jobs=-1)
                    elif model_type_select == 'SVM': base_model = SVC(C=model_params['C'], kernel=model_params['kernel'], gamma=model_params['gamma'], probability=True, class_weight=class_weights_select, random_state=42, max_iter=5000)
                    else: lr_params_run = {'C': model_params['C'], 'penalty': model_params['penalty'], 'solver': model_params['solver'], 'class_weight': class_weights_select, 'random_state': 42, 'max_iter': 5000, 'n_jobs':-1}; base_model = LogisticRegression(**lr_params_run)
                    if model_type_select == 'Voting Ensemble':
                        estimators = []
                        if use_rf:
                            rf_model = RandomForestClassifier(
                                n_estimators=100, max_depth=5, min_samples_split=4,
                                class_weight=class_weights_select, random_state=42, n_jobs=-1)
                            estimators.append(('rf', rf_model))
                        if use_svm:
                            svm_model = SVC(
                                C=1.0, kernel='rbf', gamma='scale', probability=True,
                                class_weight=class_weights_select, random_state=42)
                            estimators.append(('svm', svm_model))
                        if use_lr:
                            lr_model = LogisticRegression(
                                C=1.0, penalty='l2', solver='liblinear', class_weight=class_weights_select,
                                random_state=42, max_iter=5000, n_jobs=-1)
                            estimators.append(('lr', lr_model))
                            
                        if not estimators:
                            st.error("No models selected for ensemble. Using default Random Forest.")
                            base_model = RandomForestClassifier(
                                n_estimators=100, random_state=42, class_weight=class_weights_select)
                        else:
                            base_model = VotingClassifier(
                                estimators=estimators, voting=ensemble_voting, n_jobs=-1)
                except Exception as e: st.error(f"Model init error: {e}"); st.stop()

                # Feature Selection
                if feature_select_method != 'None':
                    st.write(f"Performing feature selection: {feature_select_method}")
                    try:
                        selector_fitted = None
                        if feature_select_method == 'Model-based Selection (RF)':
                            selector_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_train_run, y_train)
                            selector_fitted = SelectFromModel(selector_model, threshold='median', prefit=True) # Use median threshold
                        elif feature_select_method == 'Recursive Feature Elimination (RFE)':
                            if hasattr(base_model, 'feature_importances_') or hasattr(base_model, 'coef_'):
                                selector_fitted = RFECV(estimator=clone(base_model), step=1, cv=3, scoring='balanced_accuracy', min_features_to_select=1, n_jobs=-1).fit(X_train_run, y_train)
                            else: st.warning(f"RFE skipped: {model_type_select} lacks importance/coeffs."); feature_select_method = 'None'
                        if selector_fitted:
                             selected_mask_run = selector_fitted.get_support()
                             if not all(selected_mask_run):
                                 selected_names_run = [f for f, s in zip(current_feature_names_run, selected_mask_run) if s]
                                 X_train_run = X_train_run.loc[:, selected_mask_run]; X_test_run = X_test_run.loc[:, selected_mask_run] # Apply to both sets
                                 st.write(f"Selected {len(selected_names_run)} features: {', '.join(selected_names_run)}")
                             else: st.write("Feature selection removed no features.")
                    except Exception as e: st.error(f"Feature selection error: {e}"); feature_select_method = 'None'

                # Train Final Model
                st.write(f"Training final {model_type_select} on {X_train_run.shape[1]} features...")
                try:
                    model_final = clone(base_model)
                    if hasattr(model_final, 'n_jobs'):
                        model_final.n_jobs = 1  # disable parallel processing
                    model_final.fit(X_train_run, y_train)
                    st.session_state.model = model_final; 
                    st.session_state.model_type = model_type_select; 
                    st.session_state.selected_features_mask = selected_mask_run; 
                    st.session_state.selected_feature_names = selected_names_run
                    st.session_state.model_trained = True; st.success("Model trained successfully!")
                except Exception as e: 
                    st.error(f"Final model training error: {e}"); 
                    st.stop()

                # Calibration
                st.session_state.calibrated_model = None
                if use_calibration_select and hasattr(st.session_state.model, 'predict_proba'): # Check if base model supports predict_proba
                    st.write("Calibrating probabilities...")
                    try:
                        param_name = 'estimator' if version.parse(sklearn.__version__) >= version.parse('0.24.0') else 'base_estimator'
                        calib_params = {param_name: st.session_state.model, 'cv': 'prefit', 'method': 'sigmoid'}
                        calibrated_model_run = CalibratedClassifierCV(**calib_params).fit(X_test_run, y_test) # Fit on selected test set
                        st.session_state.calibrated_model = calibrated_model_run
                        st.write("Probabilities calibrated.")
                    except Exception as e: st.error(f"Calibration error: {e}")
                elif use_calibration_select:
                    st.warning("Calibration skipped: Base model does not support predict_proba.")


                # Cross-Validation (using selected features in X_train_run)
                st.write(f"Performing CV ({cv_method_select})...")
                cv_model_to_run = clone(base_model) # Use a clone of the base model for CV consistency
                try:
                    if cv_method_select == 'K-Fold (5)':
                        cv_scores = cross_val_score(cv_model_to_run, X_train_run, y_train, cv=5, scoring='balanced_accuracy', n_jobs=-1)
                        st.metric(f"K-Fold CV Balanced Accuracy", f"{cv_scores.mean():.2f} ± {cv_scores.std() * 2:.2f}")
                    elif cv_method_select == 'Leave-One-Subject-Out':
                        p_ids = st.session_state.participant_ids
                        if p_ids is None: st.warning("Participant IDs unavailable for LOGO CV.")
                        else:
                            logo = LeaveOneGroupOut(); train_indices = X_train.index; train_participant_ids = p_ids[train_indices]
                            if len(train_participant_ids) == len(X_train_run):
                                 n_splits = logo.get_n_splits(X_train_run, y_train, groups=train_participant_ids); logo_progress = st.progress(0); cv_scores = []
                                 for i, (train_idx, test_idx) in enumerate(logo.split(X_train_run, y_train, groups=train_participant_ids)):
                                     X_fold_train, X_fold_test = X_train_run.iloc[train_idx], X_train_run.iloc[test_idx]; y_fold_train, y_fold_test = y_train.iloc[train_idx], y_train.iloc[test_idx]
                                     fold_model = clone(cv_model_to_run); fold_model.fit(X_fold_train, y_fold_train); y_pred_fold = fold_model.predict(X_fold_test); cv_scores.append(balanced_accuracy_score(y_fold_test, y_pred_fold))
                                     logo_progress.progress((i + 1) / n_splits)
                                 logo_progress.empty(); cv_scores = np.array(cv_scores); st.metric(f"LOGO CV Balanced Accuracy", f"{cv_scores.mean():.2f} ± {cv_scores.std() * 2:.2f}")
                            else: st.warning("Participant ID/Training data mismatch for LOGO CV.")
                except Exception as e: st.error(f"Cross-validation error: {e}")

            st.markdown('</div>', unsafe_allow_html=True) # End training results section


        # --- Evaluation and Manual Prediction Section ---
        # Display only if model training succeeded in this session run
        if st.session_state.get('model_trained', False):
            st.markdown('<div class="results-section" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.header("📈 Model Evaluation Results")

            # Prepare evaluation data from session state
            X_test_eval = st.session_state.X_test; y_test_eval = st.session_state.y_test; scaler_eval = st.session_state.scaler
            model_eval = st.session_state.calibrated_model if st.session_state.calibrated_model else st.session_state.model
            selected_mask_eval = st.session_state.selected_features_mask; selected_names_eval = st.session_state.selected_feature_names
            model_type_eval = st.session_state.model_type

            if X_test_eval is None or y_test_eval is None or scaler_eval is None or model_eval is None:
                 st.error("Evaluation data/model missing from session state. Please retrain.")
            else:
                # Scale test data & Apply feature selection
                X_test_scaled_eval = pd.DataFrame(scaler_eval.transform(X_test_eval), columns=X_test_eval.columns, index=X_test_eval.index)
                X_test_final_eval = X_test_scaled_eval
                if selected_mask_eval is not None and not all(selected_mask_eval) and selected_names_eval:
                     try: X_test_final_eval = X_test_scaled_eval[selected_names_eval]
                     except KeyError as e: st.error(f"Feature selection mismatch during eval: {e}"); st.stop()

                # Make predictions for evaluation
                try: y_pred_eval = model_eval.predict(X_test_final_eval)
                except Exception as e: st.error(f"Error predicting on test set for eval: {e}"); st.stop()

                # Display evaluation tabs
                eval_tabs_list = st.tabs(["Report", "Confusion Matrix", "ROC Curve", "Learning Curve", "Calibration", "Influence"])
                with eval_tabs_list[0]: # Classification Report
                    st.subheader("📊 Classification Report")
                    try:
                        report = classification_report(y_test_eval, y_pred_eval, target_names=['Control', 'Aphasic'], output_dict=True, zero_division=0)
                        report_df = pd.DataFrame(report).transpose(); st.dataframe(report_df.style.format("{:.3f}"))
                        if report_df.loc['accuracy', 'f1-score'] > 0.98: st.markdown("""<div class="warning-box">⚠️ Near-perfect accuracy. Check overfitting.</div>""", unsafe_allow_html=True)
                    except Exception as e: st.error(f"Report error: {e}")
                with eval_tabs_list[1]: # Confusion Matrix
                    st.subheader("🎯 Confusion Matrix")
                    try:
                        cm = confusion_matrix(y_test_eval, y_pred_eval); fig_cm = plt.figure(figsize=(7, 5)); ax_cm = fig_cm.add_subplot(111)
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Control', 'Aphasic'], yticklabels=['Control', 'Aphasic'], ax=ax_cm)
                        ax_cm.set_title(f'CM - {model_type_eval}'); ax_cm.set_ylabel('True'); ax_cm.set_xlabel('Predicted'); st.pyplot(fig_cm); plt.close(fig_cm)
                        tn, fp, fn, tp = cm.ravel() if cm.size >= 4 else (cm[0,0] if y_test_eval.unique()[0]==0 else 0, 0, 0, cm[0,0] if y_test_eval.unique()[0]==1 else 0) if cm.size==1 else (0,0,0,0) # Handle single class case better
                        total = tn + fp + fn + tp;
                        colcm1, colcm2 = st.columns(2)
                        with colcm1: st.metric("False Positives (FP)", f"{fp} ({fp/total*100:.1f}%)" if total and (fp+tn)>0 else f"{fp}")
                        with colcm2: st.metric("False Negatives (FN)", f"{fn} ({fn/total*100:.1f}%)" if total and (fn+tp)>0 else f"{fn}")
                    except Exception as e: st.error(f"CM error: {e}")
                with eval_tabs_list[2]: # ROC Curve
                    st.subheader("📈 ROC Curve")
                    fig_roc = plot_roc_curve(model_eval, X_test_final_eval, y_test_eval, model_type_eval)
                    if fig_roc: st.pyplot(fig_roc) # Already closed in function
                with eval_tabs_list[3]: # Learning Curve
                    st.subheader("📉 Learning Curve")
                    X_lc = st.session_state.X_original; y_lc = st.session_state.y_original # Use original data
                    if selected_mask_eval is not None and not all(selected_mask_eval): X_lc = X_lc.loc[:, selected_mask_eval] # Apply selection
                    fig_lc = plot_learning_curve(st.session_state.model, X_lc, y_lc, model_type_eval) # Use uncalibrated base model
                    if fig_lc: st.pyplot(fig_lc) # Already closed in function
                    try: # Check gap
                        _, train_scores_lc, val_scores_lc = learning_curve(st.session_state.model, X_lc.values, y_lc.values, cv=5, n_jobs=-1, scoring='accuracy', error_score='raise')
                        if (np.mean(train_scores_lc, axis=1)[-1] - np.mean(val_scores_lc, axis=1)[-1]) > 0.15: st.markdown("""<div class="warning-box">⚠️ Large gap in Learning Curve suggests overfitting.</div>""", unsafe_allow_html=True)
                    except: pass
                with eval_tabs_list[4]: # Calibration Curve
                    st.subheader("📏 Probability Calibration")
                    # Need X_train_scaled (after selection) used during training
                    try: # Regenerate train split consistently
                         X_train_calib, _, y_train_calib, _ = train_test_split(st.session_state.X_original, st.session_state.y_original, test_size=0.2, random_state=42, stratify=(st.session_state.y_original if len(st.session_state.y_original.unique()) > 1 else None))
                         X_train_scaled_calib = pd.DataFrame(scaler_eval.transform(X_train_calib), columns=X_train_calib.columns, index=X_train_calib.index)
                         if selected_mask_eval is not None and not all(selected_mask_eval) and selected_names_eval: X_train_scaled_calib = X_train_scaled_calib[selected_names_eval]
                         fig_calib = plot_calibration_curve(model_eval, X_train_scaled_calib, y_train_calib, X_test_final_eval, y_test_eval)
                         if fig_calib: st.pyplot(fig_calib); st.caption("Diagonal = perfect calibration.") # Already closed in function
                    except Exception as e: st.error(f"Calibration plot generation error: {e}")
                with eval_tabs_list[5]: # Feature Influence
                    st.subheader("🎯 Feature Influence Analysis")
                    plot_feature_names_eval = selected_names_eval if selected_names_eval else st.session_state.feature_names
                    model_to_explain = st.session_state.model # Explain base model

                    if model_type_eval == 'Random Forest':
                        if hasattr(model_to_explain, 'feature_importances_'):
                            imps = model_to_explain.feature_importances_
                            if len(imps) == len(plot_feature_names_eval):
                                 fig_imp = plot_feature_selection_importance(np.ones_like(imps, dtype=bool), plot_feature_names_eval, imps)
                                 if fig_imp: st.pyplot(fig_imp) # Closed in function
                                 if len(imps)>0 and imps.max() > 0.6: st.markdown("""<div class="warning-box">⚠️ Feature Dominance? (Importance > 0.6)</div>""", unsafe_allow_html=True)
                            else: st.warning("Importance length mismatch.")
                        else: st.warning("Cannot get RF importance.")
                    elif model_type_eval in ['Logistic Regression', 'SVM'] and hasattr(model_to_explain, 'coef_'):
                         coeffs = model_to_explain.coef_[0]
                         if len(coeffs) == len(plot_feature_names_eval):
                              coef_df = pd.DataFrame({'Feature': plot_feature_names_eval, 'Coefficient': coeffs}).sort_values('Coefficient', key=abs, ascending=False)
                              fig_coef = px.bar(coef_df, x='Coefficient', y='Feature', title=f'Feature Coefficients ({model_type_eval})', color='Coefficient', color_continuous_scale=px.colors.diverging.RdBu_r, color_continuous_midpoint=0, template='plotly_dark')
                              fig_coef.update_layout(height=max(300, len(plot_feature_names_eval)*30), hovermode="y unified"); st.plotly_chart(fig_coef)
                              st.caption("Positive coeff -> Aphasic, Negative -> Control. Magnitude = Strength.")
                         else: st.warning("Coefficient length mismatch.")
                    elif model_type_eval == 'SVM' and model_to_explain.kernel != 'linear': 
                        st.info("Influence explanation complex for non-linear SVM.")
                    elif model_type_eval == 'Voting Ensemble':
                        st.write("For Voting Ensemble models, influence is shown for component models:")
                        
                        # Try to extract the first available feature importance
                        if hasattr(model_to_explain, 'estimators_'):
                            
                            if hasattr(model_to_explain, 'named_estimators_'):
                            # Find the first model with feature importances
                                for name, estimator in model_to_explain.named_estimators_.items():
                                    if hasattr(estimator, 'feature_importances_'):
                                        st.write(f"**Feature importance from {name} model:**")
                                        imps = estimator.feature_importances_
                                        if len(imps) == len(plot_feature_names_eval):
                                            fig_imp = plot_feature_selection_importance(
                                            np.ones_like(imps, dtype=bool), 
                                            plot_feature_names_eval, imps)
                                        if fig_imp: 
                                            st.pyplot(fig_imp)
                                        break
                                    elif hasattr(estimator, 'coef_'):
                                        st.write(f"**Feature coefficients from {name} model:**")
                                        coeffs = estimator.coef_[0]
                                        if len(coeffs) == len(plot_feature_names_eval):
                                            coef_df = pd.DataFrame({
                                                'Feature': plot_feature_names_eval, 
                                                'Coefficient': coeffs
                                            }).sort_values('Coefficient', key=abs, ascending=False)
                                            fig_coef = px.bar(
                                                coef_df, x='Coefficient', y='Feature', 
                                                title=f'Feature Coefficients ({name})', 
                                                color='Coefficient', color_continuous_scale=px.colors.diverging.RdBu_r, 
                                                color_continuous_midpoint=0, template='plotly_dark')
                                            fig_coef.update_layout(
                                                height=max(300, len(plot_feature_names_eval)*30), 
                                                hovermode="y unified")
                                            st.plotly_chart(fig_coef)
                                            break
                            else:
                                # For older scikit-learn versions or as fallback
                                for i, estimator in enumerate(model_to_explain.estimators_):
                                    name = f"Model {i+1}"
                                    if hasattr(estimator, 'feature_importances_'):
                                        st.write(f"**Feature importance from {name}:**")
                                        imps = estimator.feature_importances_
                                        if len(imps) == len(plot_feature_names_eval):
                                            fig_imp = plot_feature_selection_importance(
                                                np.ones_like(imps, dtype=bool), 
                                                plot_feature_names_eval, imps)
                                            if fig_imp: 
                                                st.pyplot(fig_imp)
                                            break
                                    elif hasattr(estimator, 'coef_'):
                                        st.write(f"**Feature coefficients from {name}:**")
                                        coeffs = estimator.coef_[0]
                                        if len(coeffs) == len(plot_feature_names_eval):
                                            coef_df = pd.DataFrame({
                                                'Feature': plot_feature_names_eval, 
                                                'Coefficient': coeffs
                                            }).sort_values('Coefficient', key=abs, ascending=False)
                                            fig_coef = px.bar(
                                                coef_df, x='Coefficient', y='Feature', 
                                                title=f'Feature Coefficients ({name})', 
                                                color='Coefficient', color_continuous_scale=px.colors.diverging.RdBu_r, 
                                                color_continuous_midpoint=0, template='plotly_dark')
                                            fig_coef.update_layout(
                                                height=max(300, len(plot_feature_names_eval)*30), 
                                                hovermode="y unified")
                                            st.plotly_chart(fig_coef)
                                            break
                                    else:
                                        st.info("No feature importance visualization available for the ensemble components.")
                        else:
                            st.info("Detailed influence information not available for this ensemble model.")
                    else: 
                        st.warning(f"Cannot get coeffs for {model_type_eval}.")

                    # Display the features with the most influence
                    if 'influence_data_elan' in locals() and influence_data_elan:
                        influence_df = pd.DataFrame(influence_data_elan).sort_values('Influence', ascending=False)

                        # Optional: Display raw data for debugging
                        if st.session_state.debug_mode:
                            st.write("Debug: Raw Influence Data")
                            st.dataframe(influence_df)

                        st.write("Top features contributing to the prediction:")

                        # Display the top influencing features with enhanced explanation
                        feature_count = 0
                        for _, row in influence_df.head(5).iterrows():  # Show up to 5 top features
                            if feature_count >= 3:  # Limit to 3 features by default unless debugger mode
                                if not st.session_state.debug_mode:
                                    break

                            feat, val, z, infl, direction = row['Feature'], row['Value'], row['Z-score'], row[
                                'Influence'], row['Direction']
                            mean = X_orig_elan[feat].mean()

                            # Determine if the feature has higher or lower value than the training mean
                            if z > 0:
                                value_relation = f"higher than train mean ({mean:.2f})"
                            elif z < 0:
                                value_relation = f"lower than train mean ({mean:.2f})"
                            else:
                                value_relation = f"equal to train mean ({mean:.2f})"

                            # Set color based on whether this feature pushes toward Aphasic or Control
                            push_col = "#FF6B4A" if direction == "Aphasic" else "#4CAF50"

                            # Create enhanced explanation that provides clinical context
                            explanation = ""
                            if feat == "words_per_minute":
                                if direction == "Control":
                                    explanation = "Higher speech rate often indicates better language fluency."
                                else:
                                    explanation = "The model associates this speech rate with aphasia."
                            elif feat == "grammaticality_ratio":
                                if direction == "Control":
                                    explanation = "Higher grammatical accuracy suggests intact language function."
                                else:
                                    explanation = "The model associates this grammaticality with aphasia."
                            elif feat == "total_pauses_per_minute":
                                if direction == "Aphasic":
                                    explanation = "More frequent pauses can indicate word-finding difficulties."
                                else:
                                    explanation = "Fewer pauses than expected for aphasia."
                            elif feat == "mean_pause_duration":
                                if direction == "Aphasic":
                                    explanation = "Longer pauses often relate to language processing difficulties."
                                else:
                                    explanation = "Shorter pauses than typically seen in aphasia."
                            elif feat == "filled_pause_ratio":
                                if direction == "Aphasic":
                                    explanation = "Higher proportion of filled pauses (um, uh) often seen in aphasia."
                                else:
                                    explanation = "Lower proportion of filled pauses than expected for aphasia."

                            # Add a note if feature influence doesn't align with clinical expectations
                            if not row['Expected']:
                                explanation += " (Note: This influence direction may differ from typical clinical expectations.)"

                            # Format output with improved clarity
                            st.markdown(f"""<div class="metric-card" style="border-left: 4px solid {push_col}">
                                            <h4>{feat.replace('_', ' ').title()}: {val:.2f}</h4>
                                            <p>Value is {value_relation}, Z={z:.2f}</p>
                                            <p>Pushes prediction towards: <strong>{direction}</strong></p>
                                            <p><em>{explanation}</em></p>
                                            </div>""", unsafe_allow_html=True)

                            feature_count += 1

                        # Add an explanatory note at the bottom
                        st.info("""
                            Feature influence is calculated based on how much each feature's value deviates from the training average 
                            and how strongly the model weighs that feature. The calculation considers clinical expectations about 
                            which features typically indicate control vs. aphasic speech.
                            """)
                    else:
                        st.info(f"Detailed influence explanation not available for {model_type_eval}.")

            st.markdown('</div>', unsafe_allow_html=True) # End Evaluation Section

            # --- Manual Prediction Section ---
            st.markdown('<div class="prediction-section" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.header("🔮 Predict New Sample (Manual Input)")
            X_orig_sliders = st.session_state.X_original
            feature_names_sliders = st.session_state.feature_names
            slider_vals = {}
            # Create sliders dynamically
            cols_manual = st.columns(2)
            for i, feat in enumerate(feature_names_sliders):
                 with cols_manual[i % 2]:
                      min_v = float(X_orig_sliders[feat].min()); max_v = float(X_orig_sliders[feat].max()); mean_v = float(X_orig_sliders[feat].mean())
                      # Adjust step based on range/type
                      step_v = 0.01 if max_v <= 1.0 else 0.1 if max_v < 20 else 1.0
                      format_v = "%.2f" if step_v == 0.01 else "%.1f" if step_v == 0.1 else "%.0f"
                      # Clamp default value within min/max
                      default_v = max(min_v, min(max_v, mean_v))
                      slider_vals[feat] = st.slider(f"{feat.replace('_',' ').title()}", min_v, max_v, default_v, step_v, format=format_v, key=f"slider_{feat}")

            if st.button("Predict Manual Sample", key="predict_manual_button"):
                manual_model = st.session_state.calibrated_model if st.session_state.calibrated_model else st.session_state.model
                manual_scaler = st.session_state.scaler
                manual_selected_mask = st.session_state.selected_features_mask
                manual_selected_names = st.session_state.selected_feature_names

                if manual_model and manual_scaler:
                    new_features = pd.DataFrame([slider_vals]) # Use collected slider values
                    new_features_scaled = pd.DataFrame(manual_scaler.transform(new_features[feature_names_sliders]), columns=feature_names_sliders) # Ensure order
                    features_for_manual_pred = new_features_scaled
                    if manual_selected_mask is not None and not all(manual_selected_mask) and manual_selected_names:
                         try: features_for_manual_pred = new_features_scaled[manual_selected_names]
                         except KeyError: st.error("Manual Pred Error: Feature mismatch."); st.stop()
                    try:
                         manual_pred = manual_model.predict(features_for_manual_pred)[0]
                         manual_probs = manual_model.predict_proba(features_for_manual_pred)[0]
                         st.session_state.manual_prediction_result = {'prediction': manual_pred, 'probabilities': manual_probs, 'features': new_features}
                         st.rerun() # Rerun ONLY to display manual result
                    except Exception as e: st.error(f"Manual prediction error: {e}")
                else: st.error("Model/Scaler not ready for manual prediction.")

            # Display Manual Prediction Results
            if 'manual_prediction_result' in st.session_state and st.session_state.manual_prediction_result:
                result = st.session_state.manual_prediction_result
                pred_label = "Aphasic" if result['prediction'] else "Control"; pred_color = '#FF6B4A' if result['prediction'] else '#4CAF50'
                conf_level = max(result['probabilities']); is_confident = conf_level >= 0.7; conf_icon = " ✓" if is_confident else " ⚠️"; conf_text = "High" if is_confident else "Low"
                st.markdown("---"); st.markdown("### 📋 Prediction Results (Manual Input)")
                st.markdown(f"""<div class="metric-card" style="border-left: 4px solid {pred_color};"><h3>Prediction: <span style="color:{pred_color};">{pred_label}{conf_icon}</span></h3><p>({conf_text} confidence - Prob: {conf_level:.3f})</p></div>""", unsafe_allow_html=True)
                col_prob_m1, col_prob_m2 = st.columns(2)
                with col_prob_m1: st.metric("Control Probability", f"{result['probabilities'][0]:.3f}")
                with col_prob_m2: st.metric("Aphasic Probability", f"{result['probabilities'][1]:.3f}")
                st.subheader("Feature Value Comparison");
                figs_manual = visualize_feature_distributions(st.session_state.X_original, st.session_state.y_original, result['features'], st.session_state.feature_names)
                if figs_manual:
                    cols_viz_m = st.columns(min(len(figs_manual), 3)) # Max 3 columns
                    for i, fig_viz_m in enumerate(figs_manual):
                        with cols_viz_m[i % len(cols_viz_m)]: st.plotly_chart(fig_viz_m, use_container_width=True)
                else: st.info("Could not generate distribution plots.")
                del st.session_state.manual_prediction_result # Clear result

            st.markdown('</div>', unsafe_allow_html=True) # End Manual Prediction Section


# =================================
# Predict from ELAN File Tab
# =================================
with tabs[1]:
    st.markdown('<div class="elan-upload-section">', unsafe_allow_html=True)
    st.header("📤 Upload ELAN File for Prediction")

    # Check prerequisites
    if not st.session_state.get('model_trained', False) or not st.session_state.model or not st.session_state.scaler:
        st.warning("Please train a model successfully in the 'Train Model' tab first.")
    else:
        elan_file_up_tab2 = st.file_uploader("Upload ELAN (.eaf)", type=['eaf'], key="elan_up_tab2")
        show_debug_info_elan = st.checkbox("Show ELAN detailed information", value=st.session_state.debug_mode, key="elan_debug_toggle") # Sync with global debug state

        if elan_file_up_tab2:
            eaf_path = None
            elan_pred_container = st.container() # Container to hold ELAN results

            with elan_pred_container:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.eaf') as tmp_file:
                        tmp_file.write(elan_file_up_tab2.getvalue()); eaf_path = tmp_file.name

                    st.subheader("⏳ Processing ELAN File...")
                    with st.spinner("Extracting features..."):
                        try: features_dict_raw = extract_features_from_elan(eaf_path)
                        except Exception as e: st.error(f"ELAN extraction error: {e}"); st.stop()

                    with st.spinner("Normalizing features..."):
                        features_dict_norm = normalize_feature_values(features_dict_raw, st.session_state.X_original, st.session_state.feature_scaling_factors)
                        features_df_norm = create_features_dataframe_from_dict(features_dict_norm)
                        if features_df_norm.empty: st.error("Failed to create DataFrame from ELAN features."); st.stop()

                    st.subheader("📊 Extracted & Normalized Features")
                    col_feat1, col_feat2, col_feat3 = st.columns(3)
                    feat_map = {'words_per_minute': ("WPM", col_feat1, ""), 'total_pauses_per_minute': ("Pauses/min", col_feat1, "/min"), 'grammaticality_ratio': ("Gramm. Ratio", col_feat2, ""), 'mean_pause_duration': ("Mean Pause (s)", col_feat2, " s"), 'filled_pause_ratio': ("Filled Ratio", col_feat3, "")}
                    for key, (lbl, col, unit) in feat_map.items():
                         if key in features_dict_norm: col.metric(lbl, f"{features_dict_norm[key]:.2f}{unit}")
                         else: col.metric(lbl, "N/A") # Show N/A if feature missing

                    # Debug Info (Only if checked AND training data exists)
                    if show_debug_info_elan and st.session_state.X_original is not None:
                        st.markdown('<div class="debug-section">', unsafe_allow_html=True)
                        st.subheader("🕵️ Debug: Feature Comparison & Distributions (ELAN)")
                        X_orig_debug = st.session_state.X_original; feature_names_debug = st.session_state.feature_names
                        comparison_data = [] ; unusual_features_debug = []
                        for feature in feature_names_debug: # Create comparison table
                             extracted_val = features_dict_norm.get(feature, np.nan)
                             if feature in X_orig_debug.columns:
                                 train_series = X_orig_debug[feature].dropna();
                                 if not train_series.empty:
                                     train_min, train_max, train_mean, train_std = train_series.min(), train_series.max(), train_series.mean(), train_series.std()
                                     z_score = (extracted_val - train_mean) / train_std if pd.notna(extracted_val) and train_std > 0 else 0
                                     percentile = (extracted_val - train_min) / (train_max - train_min) * 100 if pd.notna(extracted_val) and (train_max - train_min) > 0 else 0; percentile = max(0, min(100, percentile))
                                     if abs(z_score) > 2: unusual_features_debug.append((feature, z_score))
                                     comparison_data.append({'Feature': feature, 'Value': extracted_val, 'Train Min': train_min, 'Train Max': train_max, 'Train Mean': train_mean, 'Z-score': z_score, 'Percentile': percentile})
                                 else: comparison_data.append({'Feature': feature, 'Value': extracted_val, 'Train Min': 'N/A', 'Train Max': 'N/A', 'Train Mean': 'N/A', 'Z-score': 'N/A', 'Percentile': 'N/A'})
                             else: comparison_data.append({'Feature': feature, 'Value': extracted_val, 'Train Min': 'N/A', 'Train Max': 'N/A', 'Train Mean': 'N/A', 'Z-score': 'N/A', 'Percentile': 'N/A'})
                        if comparison_data: st.dataframe(pd.DataFrame(comparison_data).style.format({'Value': '{:.2f}', 'Train Min': '{:.2f}', 'Train Max': '{:.2f}', 'Train Mean': '{:.2f}', 'Z-score': '{:.2f}', 'Percentile': '{:.1f}%'}, na_rep='N/A'))
                        if unusual_features_debug: # Highlight unusual
                            st.markdown('<div class="warning-box">⚠️ Unusual Features (> 2 std dev from train mean):</div>', unsafe_allow_html=True)
                            for feature, z_score in unusual_features_debug: st.markdown(f"- **{feature}**: {abs(z_score):.2f} std dev {'above' if z_score > 0 else 'below'} mean")
                        # Distributions vs Training
                        st.write("**Feature Distributions vs. Training Data:**")
                        figs_dist_elan = visualize_feature_distributions(X_orig_debug, st.session_state.y_original, features_df_norm, feature_names_debug)
                        if figs_dist_elan:
                             cols_viz_e = st.columns(min(len(figs_dist_elan), 3))
                             for i, fig_viz_e in enumerate(figs_dist_elan):
                                 with cols_viz_e[i % len(cols_viz_e)]: st.plotly_chart(fig_viz_e, use_container_width=True)
                        else: st.info("Could not generate distribution plots.")
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Prediction
                    st.subheader("🤖 Performing Prediction...")
                    try:
                        scaler_elan = st.session_state.scaler; model_elan = st.session_state.calibrated_model if st.session_state.calibrated_model else st.session_state.model
                        features_req = st.session_state.feature_names; selected_mask_elan = st.session_state.selected_features_mask; selected_names_elan = st.session_state.selected_feature_names
                        # Ensure df has required columns in correct order for scaler
                        features_df_scaled_input = features_df_norm.reindex(columns=features_req, fill_value=features_df_norm.mean().to_dict()) # Fill missing w/ mean
                        features_scaled_elan = pd.DataFrame(scaler_elan.transform(features_df_scaled_input), columns=features_req)
                        features_for_pred_elan = features_scaled_elan
                        if selected_mask_elan is not None and not all(selected_mask_elan) and selected_names_elan: features_for_pred_elan = features_scaled_elan[selected_names_elan]

                        pred_elan = model_elan.predict(features_for_pred_elan)[0]
                        probs_elan = model_elan.predict_proba(features_for_pred_elan)[0]
                    except Exception as e: st.error(f"ELAN Prediction Error: {e}"); st.stop()

                    # Display Results
                    st.subheader("📋 Prediction Results (ELAN)")
                    pred_label_e = "Aphasic" if pred_elan else "Control"; pred_color_e = '#FF6B4A' if pred_elan else '#4CAF50'
                    conf_level_e = max(probs_elan); is_confident_e = conf_level_e >= 0.7; conf_icon_e = " ✓" if is_confident_e else " ⚠️"; conf_text_e = "High" if is_confident_e else "Low"
                    st.markdown(f"""<div class="metric-card" style="border-left: 4px solid {pred_color_e};"><h3>Prediction: <span style="color:{pred_color_e};">{pred_label_e}{conf_icon_e}</span></h3><p>({conf_text_e} confidence - Prob: {conf_level_e:.3f})</p></div>""", unsafe_allow_html=True)
                    col_prob_e1, col_prob_e2 = st.columns(2);
                    with col_prob_e1: st.metric("Control Probability", f"{probs_elan[0]:.3f}")
                    with col_prob_e2: st.metric("Aphasic Probability", f"{probs_elan[1]:.3f}")

                    # Prediction Explanation (ELAN)
                    if st.session_state.X_original is not None:
                        st.subheader("💡 Key Features Influencing Prediction (ELAN)")
                        X_orig_elan = st.session_state.X_original
                        features_to_explain_elan = selected_names_elan if selected_names_elan else st.session_state.feature_names
                        current_vals_norm_elan = features_df_norm # Use normalized values
                        model_type_elan = st.session_state.model_type
                        model_explain_elan = st.session_state.model # Base model
                        influence_data_elan = []

                        # Define clinical expectations for each feature
                        # Positive value means higher values are clinically associated with Control
                        # Negative value means higher values are clinically associated with Aphasic
                        clinical_expectations = {
                            'words_per_minute': 1.0,  # Higher WPM → Control
                            'total_pauses_per_minute': -1.0,  # More pauses → Aphasic
                            'grammaticality_ratio': 1.0,  # Higher grammaticality → Control
                            'mean_pause_duration': -1.0,  # Longer pauses → Aphasic
                            'filled_pause_ratio': -1.0  # More filled pauses → Aphasic
                        }

                        # Get class labels for clarity in the explanation
                        predicted_class = "Aphasic" if pred_elan else "Control"

                        # Calculate influence based on model type
                        if model_type_elan == 'Random Forest' and hasattr(model_explain_elan, 'feature_importances_'):
                            importances = model_explain_elan.feature_importances_
                            if len(importances) == len(features_to_explain_elan):
                                for i, feat in enumerate(features_to_explain_elan):
                                    mean, std = X_orig_elan[feat].mean(), X_orig_elan[feat].std()
                                    val = current_vals_norm_elan[feat].values[0]
                                    z = (val - mean) / std if std > 0 else 0

                                    # For Random Forest, we use importance * z-score * clinical expectation
                                    # If the feature has higher value than average, the z-score is positive
                                    # We multiply by clinical expectation to align with expected direction
                                    raw_influence = importances[i] * z

                                    # Apply clinical expectation direction
                                    if feat in clinical_expectations:
                                        clinical_direction = clinical_expectations[feat]
                                        # Determine if this feature is pushing toward Control or Aphasic
                                        # If raw_influence and clinical_direction have the same sign,
                                        # the feature pushes toward Control (when positive) or Aphasic (when negative)
                                        aligned_with_expectation = (raw_influence * clinical_direction > 0)

                                        # The influence sign determines direction: + → Control, - → Aphasic
                                        influence_direction = "Control" if (
                                                    raw_influence * clinical_direction > 0) else "Aphasic"

                                        # Store absolute importance for ranking, but preserve direction information
                                        influence_data_elan.append({
                                            'Feature': feat,
                                            'Value': val,
                                            'Z-score': z,
                                            'Influence': abs(raw_influence),  # Absolute value for ranking
                                            'Direction': influence_direction,  # Store direction separately
                                            'Expected': aligned_with_expectation,
                                            # Is this aligned with clinical expectation?
                                            'Raw': raw_influence  # Store raw value for debugging
                                        })

                            else:
                                st.warning("RF influence mismatch.")

                        elif model_type_elan in ['Logistic Regression', 'SVM'] and hasattr(model_explain_elan, 'coef_'):
                            coeffs = model_explain_elan.coef_[0]
                            if len(coeffs) == len(features_to_explain_elan):
                                for i, feat in enumerate(features_to_explain_elan):
                                    mean, std = X_orig_elan[feat].mean(), X_orig_elan[feat].std()
                                    val = current_vals_norm_elan[feat].values[0]
                                    z = (val - mean) / std if std > 0 else 0

                                    # For logistic regression, positive coefficient means higher value → Aphasic
                                    # We multiply by clinical expectation to get the right direction
                                    raw_influence = coeffs[i] * z

                                    # Apply clinical expectation direction
                                    if feat in clinical_expectations:
                                        clinical_direction = clinical_expectations[feat]

                                        # For LR/SVM, positive coef → Aphasic, so we need to flip clinical_direction
                                        # if we want higher values of "good" features to push toward Control
                                        aligned_with_expectation = (raw_influence * -clinical_direction > 0)

                                        # The influence direction for LR/SVM: + → Aphasic, - → Control
                                        influence_direction = "Aphasic" if raw_influence > 0 else "Control"

                                        influence_data_elan.append({
                                            'Feature': feat,
                                            'Value': val,
                                            'Z-score': z,
                                            'Influence': abs(raw_influence),  # Absolute value for ranking
                                            'Direction': influence_direction,  # Store direction separately
                                            'Expected': aligned_with_expectation,
                                            # Is this aligned with clinical expectation?
                                            'Raw': raw_influence  # Store raw value for debugging
                                        })
                            else:
                                st.warning("Coefficient length mismatch.")

                        elif model_type_elan == 'SVM' and model_explain_elan.kernel != 'linear':
                            st.info("Influence explanation complex for non-linear SVM.")
                        else:
                            st.warning(f"Cannot get coeffs for {model_type_elan}.")

                        # Display the features with the most influence
                        if influence_data_elan:
                            influence_df = pd.DataFrame(influence_data_elan).sort_values('Influence', ascending=False)

                            # Optional: Display raw data for debugging
                            if st.session_state.debug_mode:
                                st.write("Debug: Raw Influence Data")
                                st.dataframe(influence_df)

                            st.write("Top features contributing to the prediction:")

                            # Display the top influencing features with enhanced explanation
                            feature_count = 0
                            for _, row in influence_df.head(5).iterrows():  # Show up to 5 top features
                                if feature_count >= 3:  # Limit to 3 features by default unless debugger mode
                                    if not st.session_state.debug_mode:
                                        break

                                feat, val, z, infl, direction = row['Feature'], row['Value'], row['Z-score'], row[
                                    'Influence'], row['Direction']
                                mean = X_orig_elan[feat].mean()

                                # Determine if the feature has higher or lower value than the training mean
                                if z > 0:
                                    value_relation = f"higher than train mean ({mean:.2f})"
                                elif z < 0:
                                    value_relation = f"lower than train mean ({mean:.2f})"
                                else:
                                    value_relation = f"equal to train mean ({mean:.2f})"

                                # Set color based on whether this feature pushes toward Aphasic or Control
                                push_col = "#FF6B4A" if direction == "Aphasic" else "#4CAF50"

                                # Create enhanced explanation that provides clinical context
                                explanation = ""
                                if feat == "words_per_minute":
                                    if direction == "Control":
                                        explanation = "Higher speech rate often indicates better language fluency."
                                    else:
                                        explanation = "The model associates this speech rate with aphasia."
                                elif feat == "grammaticality_ratio":
                                    if direction == "Control":
                                        explanation = "Higher grammatical accuracy suggests intact language function."
                                    else:
                                        explanation = "The model associates this grammaticality with aphasia."
                                elif feat == "total_pauses_per_minute":
                                    if direction == "Aphasic":
                                        explanation = "More frequent pauses can indicate word-finding difficulties."
                                    else:
                                        explanation = "Fewer pauses than expected for aphasia."
                                elif feat == "mean_pause_duration":
                                    if direction == "Aphasic":
                                        explanation = "Longer pauses often relate to language processing difficulties."
                                    else:
                                        explanation = "Shorter pauses than typically seen in aphasia."
                                elif feat == "filled_pause_ratio":
                                    if direction == "Aphasic":
                                        explanation = "Higher proportion of filled pauses (um, uh) often seen in aphasia."
                                    else:
                                        explanation = "Lower proportion of filled pauses than expected for aphasia."

                                # Add a note if feature influence doesn't align with clinical expectations
                                if not row['Expected']:
                                    explanation += " (Note: This influence direction may differ from typical clinical expectations.)"

                                # Format output with improved clarity
                                st.markdown(f"""<div class="metric-card" style="border-left: 4px solid {push_col}">
                                                <h4>{feat.replace('_', ' ').title()}: {val:.2f}</h4>
                                                <p>Value is {value_relation}, Z={z:.2f}</p>
                                                <p>Pushes prediction towards: <strong>{direction}</strong></p>
                                                <p><em>{explanation}</em></p>
                                                </div>""", unsafe_allow_html=True)

                                feature_count += 1

                            # Add an explanatory note at the bottom
                            st.info("""
                                Feature influence is calculated based on how much each feature's value deviates from the training average 
                                and how strongly the model weighs that feature. The calculation considers clinical expectations about 
                                which features typically indicate control vs. aphasic speech.
                                """)
                        else:
                            st.info(f"Detailed influence explanation not available for {model_type_elan}.")

                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    if show_debug_info_elan:
                        st.code(traceback.format_exc())

                finally:
                    if eaf_path and os.path.exists(eaf_path):
                        try:
                            os.unlink(eaf_path)
                        except Exception as e_unlink: \
                    st.warning(f"Could not delete temp file: {e_unlink}")

            st.markdown('</div>', unsafe_allow_html=True)


# ===========================
#     Documentation Tab
# ===========================
with tabs[2]:
    # (Using documentation from v2 - seems comprehensive)
    st.markdown("""
        # ELAN Aphasia Classifier Documentation

        ## Overview
        This tool trains machine learning models (Random Forest, SVM, Logistic Regression) to classify speech samples based on linguistic features extracted from ELAN annotation files or provided CSVs. It aims to distinguish between typical (Control) and aphasic speech patterns.

        ## Features Used for Classification
        The classifier analyzes the following features (ensure your input data contains these or can derive them):
        * **`words_per_minute`**: Speech rate.
        * **`total_pauses_per_minute`**: Frequency of pauses (filled + unfilled).
        * **`grammaticality_ratio`**: Proportion of grammatical utterances.
        * **`mean_pause_duration`**: Average length of silent pauses (in seconds).
        * **`filled_pause_ratio`**: Proportion of total pauses that are filled (e.g., "um", "uh").

        ## How to Use

        ### 1. Train Model Tab
        * **Upload Data**: Provide two CSV files:
            * **Statistics CSV**: Must contain columns like `participant_id`, `group` ('aphasic' or 'control'), `words_per_minute`, `total_pauses`, `recording_duration_minutes`, `grammatical_utterances`, `ungrammatical_utterances`, `filled_pauses`.
            * **Pauses CSV**: Must contain `participant_id`, `group` and `duration` (of individual pauses, ideally in seconds).
        * **Automatic Scaling**: The app attempts to detect and correct obvious scaling issues (e.g., WPM < 5) during loading. Check warnings. Scaling factors are stored and used for ELAN normalization.
        * **Configure Model**: Once data is loaded, configure model type, hyperparameters, feature selection, cross-validation etc.
        * **Ensemble Options**: If you select 'Voting Ensemble' as the model type, you can:
            * Choose which base models to include (RF, SVM, LR)
            * Select voting type ('soft' uses probabilities, 'hard' uses class predictions)
            * 'Soft' voting generally performs better when models are well-calibrated
        * **Train**: Click "Train Model". Training only proceeds if data is loaded.
        * **Review**: Analyze the "Model Evaluation Results" tabs (Report, Confusion Matrix, ROC, etc.). This section appears after successful training.
        * **Predict Manually**: Use the sliders in the "Predict New Sample" section (appears after training) and click "Predict Manual Sample" to see results for custom input.

        ### 2. Predict from ELAN File Tab
        * **Prerequisite**: A model must be trained first in the "Train Model" tab.
        * **Upload ELAN**: Upload an ELAN `.eaf` file. Ensure it contains tiers compatible with the `elan_parser2.py` script used for feature extraction (check that script for expected tier names).
        * **Normalization**: Features extracted from the ELAN file are automatically normalized based on training data scale factors and heuristics to match the model's expectations. Review any "Normalization Notes".
        * **Review Prediction**: Check the "Prediction Results" card (Aphasic/Control), confidence indicator (✓ High, ⚠️ Low), and probabilities.
        * **Analyze Influence**: Review the "Key Features Influencing Prediction" section to see which feature values most strongly contributed to the classification.
        * **Debug (Optional)**: Check "Show ELAN debugging information" to see detailed feature values compared to training data ranges and distributions.

        ## Interpreting Results
        * **Prediction**: The model's classification (Aphasic/Control).
        * **Confidence**: High (✓) or Low (⚠️) based on probability threshold (0.7). Low confidence suggests the model is uncertain; interpret with caution.
        * **Probability**: The model's estimated probability for each class (calibrated if the option was selected during training).
        * **Feature Influence**: Shows which features (and their values relative to the training data) pushed the prediction towards 'Aphasic' or 'Control'.

        ## Important Notes & Potential Issues
        * **Data Quality**: The model's performance heavily depends on the quality and consistency of the training data and ELAN annotations.
        * **Scaling**: Ensure feature scales are consistent between training CSVs and ELAN extraction. The automatic normalization helps but might require adjustments based on your specific data conventions.
        * **ELAN Tier Names**: Feature extraction relies on specific tier names defined in `elan_parser2.py`. Verify these match your `.eaf` files.
        * **External Scripts**: This app depends on `elan_parser2.py` and `fixed_integration.py`. Ensure they are present and correctly implemented in the expected location.
        * **Clinical Context**: This tool provides computational analysis but is **not** a diagnostic tool. Results should always be interpreted within a broader clinical context by qualified professionals.

        ## Best Practices
        * Use representative and balanced training data.
        * Start with default model settings (e.g., Random Forest) and adjust based on evaluation results.
        * Use Leave-One-Subject-Out cross-validation if participant IDs are available for a more robust performance estimate.
        * Pay attention to warnings about overfitting, feature dominance, or unusual feature values.
        """)


# --- Main Function --- (Optional for Streamlit)
def main():
    init_session_state()
    # App logic is defined within the tab structures above
    pass

if __name__ == "__main__":
    main()