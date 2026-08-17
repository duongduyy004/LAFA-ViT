from __future__ import annotations

from favit_lsda.evaluation import build_evaluation_parser, run_evaluation


def main() -> None:
    parser = build_evaluation_parser(
        "Evaluate FA-ViT + LSDA on FaceForensics++",
        (
            "FF++ frame manifest. Defaults to data.ffpp_test_frames, then "
            "data.validation_frames"
        ),
    )
    args = parser.parse_args()
    run_evaluation(
        args,
        dataset_name="FaceForensics++",
        default_manifest_keys=("ffpp_test_frames", "validation_frames"),
    )


if __name__ == "__main__":
    main()
