from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple


def calculate_alignment_metrics(
    assertion_results: Dict[str, Dict[str, List[Tuple[Dict[str, Any], int]]]]
) -> Dict[str, Any]:
    """
    Calculate alignment metrics for each criterion and its assertions.

    Args:
        assertion_results (Dict[str, Dict[str, List[Tuple[Dict[str, Any], int]]]]):
            Nested dictionary containing assertion results.
            Structure: {criterion: {assertion: List[Tuple[score_dict, human_annotation]]}}

    Returns:
        Dict[str, Any]: A dictionary containing alignment metrics per criterion and per assertion.
    """
    metrics = {}

    # Iterate over each criterion in the assertion results
    for criterion, assertions in assertion_results.items():
        criterion_metrics = {}

        # Initialize per-criterion totals to aggregate over all assertions
        criterion_total_outputs = (
            0  # Total number of outputs across all assertions for this criterion
        )
        criterion_total_bad = 0  # Total number of bad outputs across all assertions
        criterion_total_good = 0  # Total number of good outputs across all assertions

        criterion_passes = (
            0  # Total number of outputs that passed across all assertions
        )
        criterion_fails = 0  # Total number of outputs that failed across all assertions
        criterion_fails_on_bad = 0  # Total number of bad outputs correctly failed
        criterion_fails_on_good = 0  # Total number of good outputs incorrectly failed

        # Iterate over each assertion under the current criterion
        for assertion, results in assertions.items():
            # Process each assertion's results
            total_outputs = len(results)  # Total number of outputs for this assertion

            # Count the number of bad and good outputs based on human annotations
            # Assuming human_annotation: 0 = bad output, 1 = good output
            total_bad = sum(
                1 for _, human_annotation in results if human_annotation == 0
            )
            total_good = sum(
                1 for _, human_annotation in results if human_annotation == 1
            )

            # Count the number of passes and fails based on the assertion's scoring
            passes = sum(1 for score_dict, _ in results if score_dict["score"] == 1)
            fails = (
                total_outputs - passes
            )  # Outputs that did not pass are considered failed

            # Count how many bad outputs were correctly failed
            fails_on_bad = sum(
                1
                for score_dict, human_annotation in results
                if human_annotation == 0 and score_dict["score"] == 0
            )
            # Count how many good outputs were incorrectly failed
            fails_on_good = sum(
                1
                for score_dict, human_annotation in results
                if human_annotation == 1 and score_dict["score"] == 0
            )

            # Calculate selectivity: proportion of outputs that passed
            selectivity = passes / total_outputs if total_outputs > 0 else 0.0

            # Calculate coverage: proportion of bad outputs correctly failed
            coverage = (fails_on_bad / total_bad) if total_bad > 0 else 0.0

            # Calculate False Failure Rate (FFR): proportion of good outputs incorrectly failed
            ffr = (fails_on_good / total_good) if total_good > 0 else 0.0

            # Calculate alignment using the given formula
            # Alignment measures the trade-off between coverage and FFR
            numerator = 2 * coverage * (1 - ffr)
            denominator = coverage + (1 - ffr)
            alignment = (numerator / denominator) if denominator > 0 else 0.0

            # Get the evaluation type ('llm' or 'code') for this assertion
            eval_type = results[0][0]["type"] if results else "unknown"

            # Store per-assertion metrics in the criterion's metrics dictionary
            criterion_metrics[assertion] = {
                "type": eval_type,
                "selectivity": selectivity,
                "coverage": coverage,
                "ffr": ffr,
                "alignment": alignment,
                "total_outputs": total_outputs,
                "total_good": total_good,
                "total_bad": total_bad,
                "passes": passes,
                "fails": fails,
                "fails_on_bad": fails_on_bad,
                "fails_on_good": fails_on_good,
            }

            # Aggregate counts for criterion-level metrics
            criterion_total_outputs += total_outputs
            criterion_total_bad += total_bad
            criterion_total_good += total_good
            criterion_passes += passes
            criterion_fails += fails
            criterion_fails_on_bad += fails_on_bad
            criterion_fails_on_good += fails_on_good

        # After processing all assertions under the criterion, calculate criterion-level metrics

        # Criterion-level selectivity: proportion of outputs that passed across all assertions
        criterion_selectivity = (
            (criterion_passes / criterion_total_outputs)
            if criterion_total_outputs > 0
            else 0.0
        )

        # Criterion-level coverage: proportion of bad outputs correctly failed across all assertions
        criterion_coverage = (
            (criterion_fails_on_bad / criterion_total_bad)
            if criterion_total_bad > 0
            else 0.0
        )

        # Criterion-level FFR: proportion of good outputs incorrectly failed across all assertions
        criterion_ffr = (
            (criterion_fails_on_good / criterion_total_good)
            if criterion_total_good > 0
            else 0.0
        )

        # Calculate criterion-level alignment using the aggregated coverage and FFR
        # Note:
        # - Even if individual assertions have coverage of 0 or FFR of 1 (leading to alignment of 0),
        #   the aggregated coverage and FFR across all assertions may result in values that produce a non-zero alignment.
        # - This happens because the aggregation considers the total counts, which can balance out the extremes of individual assertions.
        numerator = 2 * criterion_coverage * (1 - criterion_ffr)
        denominator = criterion_coverage + (1 - criterion_ffr)
        criterion_alignment = (numerator / denominator) if denominator > 0 else 0.0

        # Store both per-assertion and per-criterion metrics in the final metrics dictionary
        metrics[criterion] = {
            "per_assertion": criterion_metrics,
            "criterion_metrics": {
                "selectivity": criterion_selectivity,
                "coverage": criterion_coverage,
                "ffr": criterion_ffr,
                "alignment": criterion_alignment,
                "total_outputs": criterion_total_outputs,
                "total_good": criterion_total_good,
                "total_bad": criterion_total_bad,
                "passes": criterion_passes,
                "fails": criterion_fails,
                "fails_on_bad": criterion_fails_on_bad,
                "fails_on_good": criterion_fails_on_good,
            },
        }

    return metrics


def select_best_assertions(
    metrics: Dict[str, Any],
    assertion_results: Dict[str, Dict[str, List[Tuple[Dict[str, Any], int]]]],
    num_assertions_per_criterion: int = None,
) -> Dict[str, Dict[str, str]]:

    best_assertions = {}

    for criterion in assertion_results.keys():
        all_assertions = list(assertion_results[criterion].keys())

        if not num_assertions_per_criterion:
            # Intelligently select the subset of assertions that maximize the criterion's alignment score

            # Initialize variables to keep track of the best subset
            max_alignment = -1
            best_subset = []

            # Generate all possible non-empty subsets of assertions
            num_assertions = len(all_assertions)
            for r in range(1, num_assertions + 1):
                for subset in combinations(all_assertions, r):
                    # Create subset of assertion_results
                    subset_assertion_results = {
                        criterion: {
                            assertion: assertion_results[criterion][assertion]
                            for assertion in subset
                        }
                    }

                    # Calculate metrics for this subset
                    subset_metrics = calculate_alignment_metrics(
                        subset_assertion_results
                    )

                    # Get the alignment score for this criterion
                    alignment = subset_metrics[criterion]["criterion_metrics"][
                        "alignment"
                    ]

                    # If the alignment score is better, update best_subset
                    if alignment > max_alignment:
                        max_alignment = alignment
                        best_subset = subset

            # Store the best subset for this criterion
            best_assertions[criterion] = {
                assertion_name: metrics[criterion]["per_assertion"][assertion_name][
                    "type"
                ]
                for assertion_name in best_subset
            }
        else:
            # When num_assertions_per_criterion is specified, select top N assertions

            # Get per_assertion_metrics
            per_assertion_metrics = metrics[criterion]["per_assertion"]

            # Sort assertions by alignment score in descending order
            sorted_assertions = sorted(
                per_assertion_metrics.items(),
                key=lambda item: item[1]["alignment"],
                reverse=True,
            )

            # Select top N assertions per criterion
            selected_assertions = sorted_assertions[:num_assertions_per_criterion]

            # Store the selected assertions along with their types
            best_assertions[criterion] = {
                assertion_name: assertion_metrics["type"]
                for assertion_name, assertion_metrics in selected_assertions
            }

    return best_assertions


def filter_assertion_results(
    assertion_results: Dict[str, Dict[str, Any]],
    best_assertions: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, Any]]:
    filtered_results = {}

    for criterion, assertions in best_assertions.items():
        if criterion in assertion_results:
            filtered_results[criterion] = {
                assertion_name: assertion_results[criterion][assertion_name]
                for assertion_name in assertions
                if assertion_name in assertion_results[criterion]
            }
    return filtered_results


def select_best_criteria(
    metrics: Dict[str, Any],
    alignment_threshold: float,
    num_criteria: Optional[int] = None,
) -> Dict[str, Any]:
    # Sort criteria by alignment score in descending order
    sorted_criteria = sorted(
        metrics.items(),
        key=lambda x: x[1]["criterion_metrics"]["alignment"],
        reverse=True,
    )

    # Filter based on alignment threshold
    threshold_filtered = [
        (criterion, data)
        for criterion, data in sorted_criteria
        if data["criterion_metrics"]["alignment"] >= alignment_threshold
    ]

    # If num_criteria is specified, limit the number of criteria
    if num_criteria is not None:
        threshold_filtered = threshold_filtered[:num_criteria]

    # Convert back to dictionary
    best_criteria = dict(threshold_filtered)

    return best_criteria


def format_alignment_metrics(metrics):
    output = ""
    output += "| Criterion                               | Assertion                               | Type      | Alignment |\n"
    output += "|-----------------------------------------|-----------------------------------------|-----------|-----------|\n"
    for criterion, criterion_data in metrics.items():
        output += "| {} | **OVERALL**                             |           | {:.2f}     |\n".format(
            criterion[:40].ljust(40), criterion_data["criterion_metrics"]["alignment"]
        )
        for assertion, assertion_data in criterion_data["per_assertion"].items():
            output += "|                                         | {} | {} | {:.2f}     |\n".format(
                assertion[:40].ljust(40),
                assertion_data["type"].ljust(9),
                assertion_data["alignment"],
            )
    return output
