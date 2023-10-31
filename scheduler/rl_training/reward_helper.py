import logging
import pandas as pd

_logger = logging.getLogger(__name__)

def compute_rewards(scheduled_job_shared_thp_ratio, util_timeseries, coeff_cont, coeff_util):
    if len(scheduled_job_shared_thp_ratio) > 0:
        # (1) Average per-job aggregated performance degradation due to contention
        agg_perf_degrad = 0
        avg_perf_degrad = 0
        for shared_thp in scheduled_job_shared_thp_ratio:
            agg_perf_degrad += 1 - shared_thp
            _logger.debug(f'=========== [REWARD] =========== perf_degrad {1 - shared_thp} shared_thp {shared_thp}')

        avg_perf_degrad = agg_perf_degrad / len(scheduled_job_shared_thp_ratio)

        # (2) EWMA of cluster-wide GPU util
        util_ewma = utilization_term(util_timeseries)

        # Adjusting coeff_cont and coeff_util
        # to meet the optimal trade-off between job- and cluster-level performance objectives.
        reward = -coeff_cont*avg_perf_degrad + coeff_util*util_ewma
        _logger.debug(f'=========== [REWARD] =========== {reward} avg_perf_degrad {avg_perf_degrad} util_ewma {util_ewma}')
    else:
        # TODO: 0 reward when no jobs are scheduled in this round
        reward = 0
        _logger.debug(f'=========== [REWARD] =========== {reward} no jobs are scheduled in this round')

    return reward


def utilization_term(util_timeseries):
    if util_timeseries == []:
        return 0
    else:
        df = pd.DataFrame(util_timeseries)
        util_ewma_timeseries = df.ewm(span=10).mean()
        util_ewma = util_ewma_timeseries.iloc[-1][0]
        _logger.debug(f'Util Timeseries {util_timeseries} Util EWMA Timeseries {util_ewma_timeseries} Util EWMA {util_ewma}')
        return util_ewma
