"""
Standalone test script to verify feature engineering fixes.
Tests the core logic without requiring the full app context.

Run with: python test_fixes_standalone.py
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_safe_builtins_function():
    """Test 1: Verify _get_safe_builtins has all required entries."""
    print("\n=== Test 1: Unified SAFE_BUILTINS ===")

    from app.services.feature_engineering import _get_safe_builtins

    builtins = _get_safe_builtins()

    required = [
        'int', 'float', 'str', 'bool', 'list', 'dict', 'tuple', 'set', 'object',
        'len', 'sum', 'min', 'max', 'abs', 'round', 'range', 'enumerate', 'zip',
        'map', 'filter', 'sorted', 'reversed', 'any', 'all', 'isinstance', 'type',
        'np', 'pd'
    ]

    missing = [name for name in required if name not in builtins]

    if missing:
        print(f"  FAIL: Missing builtins: {missing}")
        return False

    print(f"  PASS: All {len(required)} required builtins present")
    print(f"  Builtins: {list(builtins.keys())}")
    return True


def test_has_string_literals():
    """Test 4: String literal detection function."""
    print("\n=== Test 4: String Literal Detection ===")

    from app.services.feature_engineering import _has_string_literals

    tests = [
        ("np.where(df['a'] > 0, 'yes', 'no')", True, "should detect string literals"),
        ("df['a'] + df['b']", True, "has quotes in column refs"),
        ("df.a + df.b", False, "no quotes at all"),
    ]

    all_pass = True
    for formula, expected, desc in tests:
        result = _has_string_literals(formula)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_pass = False
        print(f"  {status}: {desc} (got {result}, expected {expected})")

    return all_pass


def test_is_commented_assignment():
    """Test 4: Commented assignment detection."""
    print("\n=== Test 4: Commented Assignment Detection ===")

    from app.services.feature_engineering import _is_commented_assignment

    tests = [
        ("# df['my_col'] = something", "my_col", True, "detect commented assignment"),
        ("#df['my_col']=x", "my_col", True, "no space after #"),
        ("# This is about my_col", "my_col", False, "just a comment"),
        ("# df['other'] = x", "my_col", False, "wrong column"),
    ]

    all_pass = True
    for line, col, expected, desc in tests:
        result = _is_commented_assignment(line, col)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_pass = False
        print(f"  {status}: {desc} (got {result}, expected {expected})")

    return all_pass


def test_validation_with_out_variable():
    """Test 2: 'out' variable support in validation."""
    print("\n=== Test 2: 'out' Variable in Validation ===")

    import pandas as pd
    from app.services.feature_engineering import validate_formula

    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    # Multi-statement formula that assigns to 'out'
    formula = "temp = df['a'] + df['b']\nout = temp * 2"

    result = validate_formula(df, formula, 'test_col')

    if result.is_valid:
        print(f"  PASS: Formula with 'out' variable validated successfully")
        return True
    else:
        print(f"  FAIL: Formula with 'out' should be valid: {result.error_message}")
        return False


def test_validation_with_object_builtin():
    """Test 1b: 'object' builtin in validation."""
    print("\n=== Test 1b: 'object' Builtin in Validation ===")

    import pandas as pd
    from app.services.feature_engineering import validate_formula

    df = pd.DataFrame({'a': [1, 2, 3]})

    formula = "np.full(len(df), None, dtype=object)"

    result = validate_formula(df, formula, 'test_col')

    if result.is_valid:
        print(f"  PASS: Formula using 'object' validated successfully")
        return True
    else:
        print(f"  FAIL: Formula using 'object' should be valid: {result.error_message}")
        return False


def test_failure_notification():
    """Test 6: Failure notification in df.attrs."""
    print("\n=== Test 6: Failure Notification in df.attrs ===")

    import pandas as pd
    from app.services.feature_engineering import apply_feature_engineering

    df = pd.DataFrame({'a': [1, 2, 3]})

    # One valid, one invalid formula
    features = [
        {
            'output_column': 'valid_col',
            'formula': "df['a'] * 2",
            'source_columns': ['a']
        },
        {
            'output_column': 'invalid_col',
            'formula': "df['nonexistent'] * 2",
            'source_columns': ['nonexistent']
        }
    ]

    result_df = apply_feature_engineering(df, features, strict=False)

    if '_feature_engineering_failures' not in result_df.attrs:
        print(f"  FAIL: Failure info should be stored in df.attrs")
        return False

    failures = result_df.attrs['_feature_engineering_failures']
    if len(failures) != 1:
        print(f"  FAIL: Should have 1 failure, got {len(failures)}")
        return False

    if failures[0]['feature'] != 'invalid_col':
        print(f"  FAIL: Should identify 'invalid_col' as failed")
        return False

    print(f"  PASS: Failure notification stored correctly in df.attrs")
    print(f"  Failure count: {result_df.attrs.get('_feature_engineering_failure_count')}")
    print(f"  Success count: {result_df.attrs.get('_feature_engineering_success_count')}")
    return True


def test_execution_with_out_variable():
    """Test 2b: 'out' variable works in execution."""
    print("\n=== Test 2b: 'out' Variable in Execution ===")

    import pandas as pd
    from app.services.feature_engineering import apply_feature_engineering

    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

    features = [{
        'output_column': 'test_out',
        'formula': "temp = df['a'] + df['b']\nout = temp * 2",
        'source_columns': ['a', 'b']
    }]

    result_df = apply_feature_engineering(df, features)

    if 'test_out' not in result_df.columns:
        print(f"  FAIL: Column 'test_out' should be created")
        return False

    expected = [10, 14, 18]
    actual = list(result_df['test_out'])
    if actual != expected:
        print(f"  FAIL: Wrong values. Expected {expected}, got {actual}")
        return False

    print(f"  PASS: 'out' variable works in execution")
    return True


def test_multistatement_detection_ast():
    """Test 7: AST-based multi-statement detection doesn't flag keyword args."""
    print("\n=== Test 7: Multi-statement Detection (AST) ===")

    from app.services.feature_engineering import _is_multistatement_formula

    # This is a SINGLE expression spread across multiple lines with keyword args
    # It should NOT be detected as multi-statement
    mileage_bucket_formula = """pd.cut(
    df["Mileage"],
    bins=[-1, 50000, 100000, 150000, 200000, 300000, np.inf],
    labels=["0-50k", "50-100k", "100-150k", "150-200k", "200-300k", "300k+"]
).astype(object)"""

    if _is_multistatement_formula(mileage_bucket_formula):
        print(f"  FAIL: pd.cut formula with keyword args should NOT be multi-statement")
        return False

    # This IS a multi-statement formula
    multi_statement_formula = """
temp = df['a'] + df['b']
result = temp * 2
"""
    if not _is_multistatement_formula(multi_statement_formula):
        print(f"  FAIL: Formula with multiple assignments SHOULD be multi-statement")
        return False

    # Single assignment should be multi-statement (needs exec)
    single_assignment = "x = df['a'] + 1"
    if not _is_multistatement_formula(single_assignment):
        print(f"  FAIL: Single assignment SHOULD be multi-statement")
        return False

    # Pure expression should NOT be multi-statement
    pure_expression = "df['a'] + df['b']"
    if _is_multistatement_formula(pure_expression):
        print(f"  FAIL: Pure expression should NOT be multi-statement")
        return False

    print(f"  PASS: AST-based multi-statement detection works correctly")
    return True


def test_mileage_bucket_execution():
    """Test 8: MileageBucket-style formula executes correctly."""
    print("\n=== Test 8: MileageBucket Formula Execution ===")

    import pandas as pd
    import numpy as np
    from app.services.feature_engineering import apply_feature_engineering

    df = pd.DataFrame({
        'Mileage': [25000, 75000, 125000, 175000, 250000, 350000, None]
    })

    features = [{
        'output_column': 'MileageBucket',
        'formula': """pd.cut(
    df["Mileage"],
    bins=[-1, 50000, 100000, 150000, 200000, 300000, np.inf],
    labels=["0-50k", "50-100k", "100-150k", "150-200k", "200-300k", "300k+"]
).astype(object)""",
        'source_columns': ['Mileage']
    }]

    result_df = apply_feature_engineering(df, features)

    if 'MileageBucket' not in result_df.columns:
        print(f"  FAIL: MileageBucket column not created")
        return False

    # Check some expected values
    expected_first = "0-50k"
    actual_first = result_df['MileageBucket'].iloc[0]
    if actual_first != expected_first:
        print(f"  FAIL: Expected '{expected_first}', got '{actual_first}'")
        return False

    print(f"  PASS: MileageBucket formula executed correctly")
    print(f"  Values: {list(result_df['MileageBucket'])}")
    return True


def test_custom_variable_detection():
    """Test 9: Multi-statement formula with custom variable name."""
    print("\n=== Test 9: Custom Variable Name Detection ===")

    import pandas as pd
    import numpy as np
    from app.services.feature_engineering import apply_feature_engineering

    df = pd.DataFrame({
        'SellerType': ['Auction', 'Tower', None, 'Other'],
        'Seller': [None, None, 'IAA', None]
    })

    # Formula uses 'assignment' variable name (like AI-generated target formula)
    features = [{
        'output_column': 'AssignmentType',
        'formula': """import numpy as np
assignment = np.where(
    df["SellerType"].isin(["Auction", "Other"]),
    "Auction",
    np.where(
        df["SellerType"].isin(["Tower"]),
        "Junker",
        "Unknown"
    )
)""",
        'source_columns': ['SellerType']
    }]

    result_df = apply_feature_engineering(df, features)

    if 'AssignmentType' not in result_df.columns:
        print(f"  FAIL: AssignmentType column not created")
        return False

    expected = ['Auction', 'Junker', 'Unknown', 'Auction']
    actual = list(result_df['AssignmentType'])
    if actual != expected:
        print(f"  FAIL: Expected {expected}, got {actual}")
        return False

    print(f"  PASS: Custom variable name 'assignment' auto-detected")
    print(f"  Values: {actual}")
    return True


def test_bare_expression_capture():
    """Test 10: Multi-statement formula with bare expression (no assignment)."""
    print("\n=== Test 10: Bare Expression Capture ===")

    import pandas as pd
    import numpy as np
    from app.services.feature_engineering import apply_feature_engineering

    df = pd.DataFrame({
        'AssignedBy': ['RKaye', 'gsavin', 'System', 'JohnDoe', 'JaneSmith']
    })

    # Formula that has a bare expression at the end (no assignment)
    # This is the pattern that was failing for AssignedByBucket
    features = [{
        'output_column': 'AssignedByBucket',
        'formula': """major_assignors = ["RKaye", "gsavin", "System"]
np.where(df["AssignedBy"].isin(major_assignors), df["AssignedBy"], "OtherAssignee")""",
        'source_columns': ['AssignedBy']
    }]

    result_df = apply_feature_engineering(df, features)

    if 'AssignedByBucket' not in result_df.columns:
        print(f"  FAIL: AssignedByBucket column not created")
        return False

    expected = ['RKaye', 'gsavin', 'System', 'OtherAssignee', 'OtherAssignee']
    actual = list(result_df['AssignedByBucket'])
    if actual != expected:
        print(f"  FAIL: Expected {expected}, got {actual}")
        return False

    print(f"  PASS: Bare expression result auto-captured")
    print(f"  Values: {actual}")
    return True


def main():
    print("=" * 60)
    print("Feature Engineering Fixes - Standalone Tests")
    print("=" * 60)

    tests = [
        test_safe_builtins_function,
        test_has_string_literals,
        test_is_commented_assignment,
        test_validation_with_object_builtin,
        test_validation_with_out_variable,
        test_execution_with_out_variable,
        test_failure_notification,
        test_multistatement_detection_ast,
        test_mileage_bucket_execution,
        test_custom_variable_detection,
        test_bare_expression_capture,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
