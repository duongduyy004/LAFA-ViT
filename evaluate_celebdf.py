from __future__ import annotations

from favit_lsda.evaluation import build_evaluation_parser, run_evaluation


def main() -> None:
    parser = build_evaluation_parser(
        "Evaluate FA-ViT + LSDA on Celeb-DF-v2",
        "Celeb-DF frame manifest (default: data.celebdf_test_frames)",
    )
    args = parser.parse_args()
    run_evaluation(
        args,
        dataset_name="Celeb-DF-v2",
        default_manifest_keys=("celebdf_test_frames",),
    )


if __name__ == "__main__":
    main()
