from typing import Dict, List, Tuple, Any, Optional
from itertools import combinations

def calculate_alignment_metrics(assertion_results: Dict[str, Dict[str, List[Tuple[Dict[str, Any], int]]]]) -> Dict[str, Any]:
    metrics = {}

    for criterion, assertions in assertion_results.items():
        criterion_metrics = {}
        # Initialize per-criterion totals
        criterion_total_outputs = 0
        criterion_total_bad = 0
        criterion_total_good = 0

        criterion_passes = 0
        criterion_fails = 0
        criterion_fails_on_bad = 0
        criterion_fails_on_good = 0

        for assertion, results in assertions.items():
            # Process each assertion
            total_outputs = len(results)

            # Assuming human_annotation: 0 = bad, 1 = good
            total_bad = sum(1 for _, human_annotation in results if human_annotation == 0)
            total_good = sum(1 for _, human_annotation in results if human_annotation == 1)

            passes = sum(1 for score_dict, _ in results if score_dict['score'] == 1)
            fails = total_outputs - passes

            fails_on_bad = sum(1 for score_dict, human_annotation in results if human_annotation == 0 and score_dict['score'] == 0)
            fails_on_good = sum(1 for score_dict, human_annotation in results if human_annotation == 1 and score_dict['score'] == 0)

            selectivity = passes / total_outputs if total_outputs > 0 else 0.0

            # Coverage: Proportion of bad outputs that are correctly failed
            coverage = (fails_on_bad / total_bad) if total_bad > 0 else 0.0

            # FFR: Proportion of good outputs that are incorrectly failed
            ffr = (fails_on_good / total_good) if total_good > 0 else 0.0

            # Calculate alignment
            numerator = 2 * coverage * (1 - ffr)
            denominator = coverage + (1 - ffr)
            alignment = (numerator / denominator) if denominator > 0 else 0.0

            eval_type = results[0][0]['type'] if results else "unknown"

            # Store per-assertion metrics
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
                "fails_on_good": fails_on_good
            }

            # Aggregate totals for criterion-level metrics
            criterion_total_outputs += total_outputs
            criterion_total_bad += total_bad
            criterion_total_good += total_good
            criterion_passes += passes
            criterion_fails += fails
            criterion_fails_on_bad += fails_on_bad
            criterion_fails_on_good += fails_on_good

        # Compute per-criterion metrics
        criterion_selectivity = (criterion_passes / criterion_total_outputs) if criterion_total_outputs > 0 else 0.0

        criterion_coverage = (criterion_fails_on_bad / criterion_total_bad) if criterion_total_bad > 0 else 0.0

        criterion_ffr = (criterion_fails_on_good / criterion_total_good) if criterion_total_good > 0 else 0.0

        # Calculate criterion-level alignment
        numerator = 2 * criterion_coverage * (1 - criterion_ffr)
        denominator = criterion_coverage + (1 - criterion_ffr)
        criterion_alignment = (numerator / denominator) if denominator > 0 else 0.0

        # Store both per-assertion and per-criterion metrics
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
                "fails_on_good": criterion_fails_on_good
            }
        }

    return metrics

def select_best_assertions(
    metrics: Dict[str, Any],
    assertion_results: Dict[str, Dict[str, List[Tuple[Dict[str, Any], int]]]],
    num_assertions_per_criterion: int = None
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
                    subset_metrics = calculate_alignment_metrics(subset_assertion_results)

                    # Get the alignment score for this criterion
                    alignment = subset_metrics[criterion]['criterion_metrics']['alignment']

                    # If the alignment score is better, update best_subset
                    if alignment > max_alignment:
                        max_alignment = alignment
                        best_subset = subset

            # Store the best subset for this criterion
            best_assertions[criterion] = {
                assertion_name: metrics[criterion]['per_assertion'][assertion_name]['type']
                for assertion_name in best_subset
            }
        else:
            # When num_assertions_per_criterion is specified, select top N assertions

            # Get per_assertion_metrics
            per_assertion_metrics = metrics[criterion]['per_assertion']

            # Sort assertions by alignment score in descending order
            sorted_assertions = sorted(
                per_assertion_metrics.items(),
                key=lambda item: item[1]['alignment'],
                reverse=True
            )

            # Select top N assertions per criterion
            selected_assertions = sorted_assertions[:num_assertions_per_criterion]

            # Store the selected assertions along with their types
            best_assertions[criterion] = {
                assertion_name: assertion_metrics['type']
                for assertion_name, assertion_metrics in selected_assertions
            }

    return best_assertions

def filter_assertion_results(
    assertion_results: Dict[str, Dict[str, Any]],
    best_assertions: Dict[str, Dict[str, str]]
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
    num_criteria: Optional[int] = None
) -> Dict[str, Any]:
    # Sort criteria by alignment score in descending order
    sorted_criteria = sorted(
        metrics.items(),
        key=lambda x: x[1]['criterion_metrics']['alignment'],
        reverse=True
    )
    
    # Filter based on alignment threshold
    threshold_filtered = [
        (criterion, data) for criterion, data in sorted_criteria
        if data['criterion_metrics']['alignment'] >= alignment_threshold
    ]
    
    # If num_criteria is specified, limit the number of criteria
    if num_criteria is not None:
        threshold_filtered = threshold_filtered[:num_criteria]
    
    # Convert back to dictionary
    best_criteria = dict(threshold_filtered)
    
    return best_criteria

def format_alignment_metrics(metrics):
    output = ""
    output += "{:<40} {:<40} {:<10} {:<10}\n".format("Criterion", "Assertion", "Type", "Alignment")
    output += "-" * 100 + "\n"
    for criterion, criterion_data in metrics.items():
        output += "{:<40} {:<40} {:<10} {:<10.2f}\n".format(
            criterion[:40],
            "OVERALL",
            "",
            criterion_data['criterion_metrics']['alignment']
        )
        output += "-" * 100 + "\n"
        for assertion, assertion_data in criterion_data['per_assertion'].items():
            output += "{:<40} {:<40} {:<10} {:<10.2f}\n".format(
                "",
                assertion[:40],
                assertion_data['type'],
                assertion_data['alignment']
            )
        output += "-" * 100 + "\n"
    return output