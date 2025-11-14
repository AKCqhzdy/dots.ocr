from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.metrics import Meter, get_meter_provider, set_meter_provider
from opentelemetry.metrics._internal import NoOpMeter
from opentelemetry.sdk.metrics import MeterProvider

_meter: Meter = NoOpMeter("")


def setup_metrics(enable_metrics: bool) -> Meter:
    """Setup OpenTelemetry metrics with OTLP exporter."""

    global _meter

    if enable_metrics:
        # Exporter to export metrics to Prometheus
        reader = PrometheusMetricReader(False)
        provider = MeterProvider(metric_readers=[reader])
        set_meter_provider(provider)
        _meter = get_meter_provider().get_meter(__name__)

    return _meter


def get_meter() -> Meter:
    """Get the global meter instance."""
    return _meter
