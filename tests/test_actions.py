from unittest.mock import MagicMock

from hungercue.telephony.actions import TwilioCallActions, transfer_twiml, voicemail_twiml


def test_twiml() -> None:
    assert transfer_twiml("+1212", "") == "<Response><Dial>+1212</Dial></Response>"
    assert transfer_twiml("+1212", "One moment").startswith("<Response><Say>One moment</Say>")
    vm = voicemail_twiml("Leave a message & hang up", "https://x/voicemail")
    assert "&amp;" in vm and 'recordingStatusCallback="https://x/voicemail"' in vm and "<Record" in vm


async def test_actions_update_live_call() -> None:
    a = TwilioCallActions("AC" + "0" * 32, "tok")
    a.client = MagicMock()
    await a.hangup("CA1")
    a.client.calls.assert_called_with("CA1")
    a.client.calls.return_value.update.assert_called_with(status="completed")
    await a.transfer("CA1", "+1212", "")
    assert "<Dial>+1212</Dial>" in a.client.calls.return_value.update.call_args.kwargs["twiml"]
    await a.voicemail("CA1", "hi", "https://x/vm")
    assert "<Record" in a.client.calls.return_value.update.call_args.kwargs["twiml"]
