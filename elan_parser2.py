import sys
import pympi # Import pympi-ling library
import math

## --- CONFIGURATION --- ##
# !!! IMPORTANT: Adjust these values to match YOUR ELAN file structure !!!

# Tier names
TIER_NAME_WORDS = 'Words' # Tier containing word/token annotations
TIER_NAME_PAUSES = 'Events'       # Tier containing explicit silent pause annotations (e.g., 'sil')
TIER_NAME_FILLED_GAPS = 'Events' # Tier containing filled pause annotations (e.g., 'FP', 'um', 'uh')
TIER_NAME_LENGTHENING = 'Events' # Tier containing lengthening annotations (e.g., 'a:', 'o:', 'i:')
TIER_NAME_GRAMMAR = 'Gramm clauses' # Tier containing utterance annotations marked for grammaticality

# Annotation values / markers
FILLED_GAP_MARKERS = ['filled gaps', 'silence', 'laugh', 'sigh', 'breath', 'noise'] # List of annotations considered filled gaps
LENGTHENING_MARKERS = ['a-lengthening', 'o-lengthening', 'i-lengthening'] # List of annotations considered lengthenings
GRAMMATICAL_MARKER = 'grammatical' # Annotation value for grammatical utterances
UNGRAMMATICAL_MARKER = 'ungrammatical' # Annotation value for ungrammatical utterances

# Thresholds
SILENT_PAUSE_THRESHOLD_SEC = 0.5 # Minimum duration for a silent pause to be counted

## --- END CONFIGURATION --- ##


def extract_features_from_elan(eaf_path):
    """
    Extracts linguistic features from an ELAN EAF file.

    Args:
        eaf_path (str): Path to the ELAN (.eaf) file.

    Returns:
        dict: A dictionary containing the extracted features,
              or None if the file cannot be processed.
    """
    try:
        eaf = pympi.Elan.Eaf(eaf_path)
    except FileNotFoundError:
        print(f"Error: File not found at {eaf_path}")
        return None
    except Exception as e:
        print(f"Error opening or parsing EAF file {eaf_path}: {e}")
        return None

    # --- Get Recording Duration (Corrected Method) ---
    # Find the maximum end time among all annotations across all tiers
    duration_ms = 0
    all_tier_names = eaf.get_tier_names()
    if not all_tier_names:
        print(f"Warning: No tiers found in EAF file {eaf_path}. Cannot determine duration.")
        # Depending on requirements, you might return None or default features
        # return None
    else:
        for tier_id in all_tier_names:
            try:
                # Annotations are typically [(start_ms, end_ms, value), ...]
                annotations = eaf.get_annotation_data_for_tier(tier_id)
                if annotations:
                    # Find the maximum end time (index 1) in this tier's annotations
                    tier_max_time = max(ann[1] for ann in annotations)
                    # Update the overall maximum duration if this tier's max is greater
                    duration_ms = max(duration_ms, tier_max_time)
            except KeyError:
                # This might happen if a tier listed doesn't actually have data, less common
                print(f"Warning: Could not retrieve annotation data for tier '{tier_id}'. Skipping for duration calculation.")
                continue # Skip to the next tier

    if duration_ms == 0:
        # This might happen if the EAF file is empty or has no timed annotations
        print(f"Warning: Could not determine recording duration (max timestamp is 0) for {eaf_path}.")
        # Decide how to handle this - return None, return default features, or continue with duration 0?
        # For calculations like per minute rates, duration 0 will cause issues.
        # Let's return None for safety.
        return None

    duration_sec = duration_ms / 1000.0
    duration_min = duration_sec / 60.0

    # --- Initialize Counters ---
    word_count = 0
    grammatical_count = 0
    ungrammatical_count = 0
    silent_pause_count = 0
    silent_pause_durations = []
    filled_gap_count = 0
    lengthening_count = 0

    # --- Process Tiers ---

    # 1. Word Count (from TIER_NAME_WORDS)
    if TIER_NAME_WORDS in eaf.get_tier_names():
        word_annotations = eaf.get_annotation_data_for_tier(TIER_NAME_WORDS)
        # Assuming each annotation is a single word/token
        word_count = len(word_annotations)
    else:
        print(f"Warning: Word tier '{TIER_NAME_WORDS}' not found.")

    # 2. Grammaticality (from TIER_NAME_GRAMMAR)
    if TIER_NAME_GRAMMAR in eaf.get_tier_names():
        grammar_annotations = eaf.get_annotation_data_for_tier(TIER_NAME_GRAMMAR)
        for _start, _end, value in grammar_annotations:
            if value == GRAMMATICAL_MARKER:
                grammatical_count += 1
            elif value == UNGRAMMATICAL_MARKER:
                ungrammatical_count += 1
    else:
        print(f"Warning: Grammar tier '{TIER_NAME_GRAMMAR}' not found.")

    # 3. Silent Pauses > threshold (from TIER_NAME_PAUSES)
    if TIER_NAME_PAUSES in eaf.get_tier_names():
        pause_annotations = eaf.get_annotation_data_for_tier(TIER_NAME_PAUSES)
        for start, end, _value in pause_annotations:
            duration = (end - start) / 1000.0 # Convert ms to sec
            if duration > SILENT_PAUSE_THRESHOLD_SEC:
                silent_pause_count += 1
                silent_pause_durations.append(duration)
    else:
        print(f"Warning: Silent Pause tier '{TIER_NAME_PAUSES}' not found.")
        # Alternative: Calculate silent pauses from gaps in TIER_NAME_WORDS
        # This is more complex and not implemented here.

    # 4. Filled Gaps (from TIER_NAME_FILLED_GAPS)
    if TIER_NAME_FILLED_GAPS in eaf.get_tier_names():
        filled_gap_annotations = eaf.get_annotation_data_for_tier(TIER_NAME_FILLED_GAPS)
        for _start, _end, value in filled_gap_annotations:
            if value in FILLED_GAP_MARKERS:
                filled_gap_count += 1
    else:
        print(f"Warning: Filled Gap tier '{TIER_NAME_FILLED_GAPS}' not found.")

    # 5. Lengthenings (from TIER_NAME_LENGTHENING)
    if TIER_NAME_LENGTHENING in eaf.get_tier_names():
        lengthening_annotations = eaf.get_annotation_data_for_tier(TIER_NAME_LENGTHENING)
        for _start, _end, value in lengthening_annotations:
            if value in LENGTHENING_MARKERS:
                lengthening_count += 1
    else:
        print(f"Warning: Lengthening tier '{TIER_NAME_LENGTHENING}' not found.")


    # --- Calculate Features ---

    # Words per minute
    words_per_minute = (word_count / duration_sec * 60) if duration_sec > 0 else 0

    # Total Pauses (Silent > threshold + Filled + Lengthening)
    total_pause_events = silent_pause_count + filled_gap_count + lengthening_count

    # Total Pauses per minute
    total_pauses_per_minute = (total_pause_events / duration_sec * 60) if duration_sec > 0 else 0

    # Grammaticality Ratio
    total_utterances = grammatical_count + ungrammatical_count
    grammaticality_ratio = (grammatical_count / total_utterances) if total_utterances > 0 else 0

    # Mean Pause Duration (only considering explicitly annotated silent pauses > threshold)
    # Note: This only includes silent pauses found on TIER_NAME_PAUSES > threshold.
    # If you want to include filled/lengthening durations, the logic needs modification.
    mean_pause_duration_sec = (sum(silent_pause_durations) / silent_pause_count) if silent_pause_count > 0 else 0

    # Filled Pause Ratio (Filled Gaps + Lengthenings) / Total Pause Events
    filled_pause_events = filled_gap_count + lengthening_count
    filled_pause_ratio = (filled_pause_events / total_pause_events) if total_pause_events > 0 else 0

    # --- Prepare Results ---
    features = {
        "participant_id": eaf_path.split('/')[-1].split('.')[0], # Basic ID extraction
        "words_per_minute": round(words_per_minute, 2),
        "total_pauses_per_minute": round(total_pauses_per_minute, 2),
        "grammaticality_ratio": round(grammaticality_ratio, 2),
        "mean_pause_duration": round(mean_pause_duration_sec, 2), # In seconds
        "filled_pause_ratio": round(filled_pause_ratio, 2),
        "recording_duration_minutes": round(duration_min, 4) # Higher precision for duration
    }

    return features

# --- Main execution block ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <path_to_elan_file.eaf>")
        sys.exit(1) # Exit with an error code

    elan_file_path = sys.argv[1]
    extracted_features = extract_features_from_elan(elan_file_path)

    if extracted_features:
        print("\nExtracted features:")
        for key, value in extracted_features.items():
            print(f"{key}: {value}")