import json
import numpy as np
import scipy
import nltk
from nltk import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


def compute_all_pairwise_scores(src_data, tgt_data, metric):
    X = np.empty((len(src_data), len(tgt_data)))

    for i in range(len(src_data)):
        for j in range(len(tgt_data)):
            X[i][j] = (metric(src_data[i], tgt_data[j]))

    return X


def pairwise_metric(candidate, reference):  # Todo this is not thread safe, no idea how to make it so
    return nltk.translate.meteor_score.single_meteor_score(word_tokenize(reference), word_tokenize(candidate))


def compute_pairwise_evidence_score(gold_evidence, pred_evidence):
    pairwise_scores = compute_all_pairwise_scores(pred_evidence, gold_evidence, pairwise_metric)
    assignment = scipy.optimize.linear_sum_assignment(pairwise_scores, maximize=True)
    assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()

    # Reweight to account for unmatched target questions
    reweight_term = 1 / float(len(gold_evidence))
    assignment_utility *= reweight_term

    return assignment_utility


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    print("Submission related metadata:")
    """
    Evaluates the submission for a particular challenge phase adn returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            "status": u"running",
            "when_made_public": None,
            "participant_team": 5,
            "input_file": "https://abc.xyz/path/to/submission/file.json",
            "execution_time": u"123",
            "publication_url": u"ABC",
            "challenge_phase": 1,
            "created_by": u"ABC",
            "stdout_file": "https://abc.xyz/path/to/stdout/file.json",
            "method_name": u"Test",
            "stderr_file": "https://abc.xyz/path/to/stderr/file.json",
            "participant_team_name": u"Test Team",
            "project_url": u"http://foo.bar",
            "method_description": u"ABC",
            "is_public": False,
            "submission_result_file": "https://abc.xyz/path/result/file.json",
            "id": 123,
            "submitted_at": u"2017-03-20T19:22:03.880652Z",
        }
    """
    print(kwargs["submission_metadata"])
    output = {}

    # load data
    gold_samples = json.load(open(test_annotation_file, 'r'))
    pred_samples = json.load(open(user_submission_file, 'r'))

    assert len(gold_samples) == len(pred_samples)

    # evaluate
    averitec_reporting_levels = [0.2, 0.25, 0.3]
    gold_labels, pred_labels, evid_scores, scores = [], [], [], []
    for test_sample, pred_sample in zip(gold_samples, pred_samples):
        gold_label, gold_evidence = test_sample['label'], test_sample['evidence']
        pred_label, pred_evidence = pred_sample['label'], pred_sample['evidence']

        # label
        gold_labels.append(gold_label)
        pred_labels.append(pred_label)

        # evidence
        score = compute_pairwise_evidence_score(gold_evidence, pred_evidence)
        evid_scores.append(score)

        # meteor
        this_example_scores = [0.0 for _ in averitec_reporting_levels]
        for i, level in enumerate(averitec_reporting_levels):
            if score > level:
                this_example_scores[i] = gold_label == pred_label

        scores.append(this_example_scores)

    # label accuracy
    label_accuracy = np.mean([s == t for s, t in zip(gold_labels, pred_labels)])
    # evidence score
    evidence_score = np.mean(evid_scores)
    # averitec score
    averitec_scores = np.mean(np.array(scores), axis=0)

    if phase_codename == "dev":
        print("Evaluating for Dev Phase")
        output["result"] = [
            {
                "dev_split": {
                    "Label accuracy": label_accuracy,
                    "Evidence score": evidence_score,
                    "Averitec score1": averitec_scores[0],  # (meteor @ 0.2)
                    "Averitec score2": averitec_scores[1],  # (meteor @ 0.25)
                    "Averitec score3": averitec_scores[2],  # (meteor @ 0.3)
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["dev_split"]
        print("Completed evaluation for Dev Phase")
    elif phase_codename == "test" or phase_codename == "after_test":
        print("Evaluating for Test Phase")
        output["result"] = [
            {
                "test_split": {
                    "Label accuracy": label_accuracy,
                    "Evidence score": evidence_score,
                    "Averitec score1": averitec_scores[0],  # (meteor @ 0.2)
                    "Averitec score2": averitec_scores[1],  # (meteor @ 0.25)
                    "Averitec score3": averitec_scores[2],  # (meteor @ 0.3)
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["test_split"]
        print("Completed evaluation for Test Phase")

    return output
