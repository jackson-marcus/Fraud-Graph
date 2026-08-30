"""Pattern #2 — Producers → Stream → Consumer Workers → Sink."""

from fraudgraph.streams.consumer import StreamConsumer
from fraudgraph.streams.producer import InMemoryStream, StreamProducer
from fraudgraph.streams.schemas import REQUIRED_FIELDS, STREAM_NAME, StreamEvent
from fraudgraph.workers.processor import FeatureProcessor


def _payload(tag: str) -> dict:
    payload = {name: (1 if name != REQUIRED_FIELDS[0] else tag) for name in REQUIRED_FIELDS}
    first = REQUIRED_FIELDS[0]
    payload[first] = tag
    return payload


def _bus():
    stream = InMemoryStream()
    return StreamProducer(stream), StreamConsumer(stream), FeatureProcessor()


def test_schema_rejects_incomplete_payloads():
    try:
        StreamEvent.create({})
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected missing-field error")


def test_producer_consumer_roundtrip():
    producer, consumer, processor = _bus()
    published = producer.publish(_payload("one"))
    assert producer.stream_name == STREAM_NAME
    handled = consumer.run_once(processor.handle)
    assert handled == 1
    assert published.event_id in consumer.processed
    assert processor.seen == 1


def test_dead_letter_on_handler_failure():
    producer, consumer, _processor = _bus()
    producer.publish(_payload("bad"))

    def boom(_event):
        raise RuntimeError("worker failed")

    consumer.run_once(boom)
    assert len(consumer.dead_letters) == 1
    assert consumer.processed == []
