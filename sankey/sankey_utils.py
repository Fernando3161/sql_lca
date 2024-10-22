import pandas as pd
from collections import defaultdict
import warnings
import matplotlib

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)


# Shorten the text to the first 10 characters for all cells
# df = df.applymap(lambda x: x[:20] if isinstance(x, str) else x)
def shorten_and_suffix(series):
    """
    Shortens strings in the series to the first 20 characters
    and adds a suffix to duplicates to ensure uniqueness.

    Parameters:
    series (pd.Series): A pandas Series that may contain strings.

    Returns:
    list: A list of modified values from the series.
    """
    seen = {}  # Dictionary to track occurrences of shortened values
    result = []

    for value in series:
        if isinstance(value, str):
            short_value = value[:20]  # Shorten to 20 characters
            if short_value in seen:
                seen[short_value] += 1
                result.append(
                    f"{short_value}_{seen[short_value]}"
                )  # Add suffix for duplicates
            else:
                seen[short_value] = 1
                result.append(short_value)  # Add shortened value
        else:
            result.append(value)  # Leave non-string values unchanged

    return result


def calculate_level(row):
    """
    Calculate the level based on the number of consecutive 'X' values
    before the first non-'X' value in a row.

    Parameters:
    row (pd.Series): A row of the DataFrame.

    Returns:
    int: The count of consecutive 'X' values.
    """
    level = 0
    for value in row:
        if value == "X":
            level += 1  # Increment level for each 'X'
        else:
            break  # Stop counting at the first non-'X' value
    return level


def create_source(df):
    """
    Create a list of source indices based on the DataFrame index.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.

    Returns:
    list: A list of indices representing the source nodes.
    """
    return df.index.tolist()  # Source is simply the index


def create_target(df):
    """
    Create a list of target indices based on the 'Level' column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the 'Level' column.

    Returns:
    list: A list of indices representing the target nodes.
    """
    target = []
    for i in range(len(df)):
        current_level = df["Level"].iloc[i]

        # Check if it's the first row
        if i == 0:
            target.append(None)  # No target for the first row
            continue

        previous_level = df["Level"].iloc[i - 1]

        if current_level == previous_level + 1:
            # Connect to the previous row
            target.append(i - 1)  # Target is the previous row index
        else:
            # Find the closest previous row with a lower level
            found_target = None
            for j in range(i - 1, -1, -1):  # Start from the previous row
                if df["Level"].iloc[j] < current_level:
                    found_target = j
                    break
            target.append(found_target)

    return target


def extract_flows(row):
    """
    Extract the first non-'X' value from a row to determine the flow name.

    Parameters:
    row (pd.Series): A row of the DataFrame containing flow values.

    Returns:
    str or None: The first non-'X' value found in the row, or None if all values are 'X'.
    """
    for value in row:
        if value != "X":
            return value
    return None


def shorten_labels(labels, max_length=None):
    """
    Shortens labels to a specified maximum length and ensures uniqueness.

    Args:
        labels (list): List of labels to shorten.
        max_length (int): Maximum length for each label.

    Returns:
        list: List of shortened labels.
    """
    if max_length is None:
        max_length = 500  # Avoid shortening if the option is disabled

    seen = defaultdict(int)  # Dictionary to count occurrences
    shortened = []

    for label in labels:
        # Truncate the label
        base_label = label[:max_length]
        seen[base_label] += 1

        # Create a unique label
        unique_label = (
            f"{base_label}_{seen[base_label] - 1}"
            if seen[base_label] > 1
            else base_label
        )
        shortened.append(unique_label)

    return shortened
