#!/usr/bin/env python3

import click
import logging
from pipeline.pipeline import BoxingDynamicsPipeline
from pipeline.video_loader import VideoLoader


@click.command()
@click.option('--debug-logging', is_flag=True, help="Enable DEBUG logging", default=False)
def main(debug_logging: bool):

    log_level = logging.DEBUG if debug_logging else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='[%(levelname)s] %(name)s: %(message)s'
    )

    logging.info("Starting BoxingDynamics pipeline")

    logging.info("Finished BoxingDynamics pipeline")


if __name__=='__main__':
    main()