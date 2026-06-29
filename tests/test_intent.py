"""Tests for the pre-retrieval intent router (offline, no LLM)."""
from __future__ import annotations

import pytest

from safety.intent import CHITCHAT, QA, SYMPTOM, chitchat_reply, classify_intent


@pytest.mark.parametrize(
    "query",
    [
        "hi",
        "hello there",
        "hey, can i ask something else?",
        "thanks!",
        "are you a bot",
        "what can you do",
        "",
        "   ?!  ",
    ],
)
def test_greetings_and_meta_route_to_chitchat(query):
    # Even with the Symptom toggle on, pleasantries get the redirect.
    assert classify_intent(query, SYMPTOM) == CHITCHAT


@pytest.mark.parametrize(
    "query",
    [
        "what is indigestion",
        "what's GERD",
        "what are the symptoms of anemia",
        "define hypertension",
        "tell me about diabetes",
        "what does HbA1c mean",
    ],
)
def test_definition_questions_route_to_qa_even_in_symptom_mode(query):
    # This is the fix for "what is Indigestion" producing a fake differential.
    assert classify_intent(query, SYMPTOM) == QA


@pytest.mark.parametrize(
    "query",
    [
        "i have a cold and pain in my stomach",
        "my head hurts and i feel dizzy",
        "i've had a fever and a cough for two days",
    ],
)
def test_first_person_symptoms_route_to_symptom_even_in_qa_mode(query):
    assert classify_intent(query, QA) == SYMPTOM


def test_no_strong_signal_follows_the_mode_hint():
    # A neutral noun phrase with no definition/treatment/symptom signal falls
    # through to the UI mode hint. ("anemia treatment options" no longer works
    # here — "treatment" is now a strong QA signal, routed to QA in both modes.)
    assert classify_intent("anemia in older adults", QA) == QA
    assert classify_intent("anemia in older adults", SYMPTOM) == SYMPTOM


def test_greeting_prefix_with_real_question_is_not_chitchat():
    # "hi what is diabetes" has a real ask -> definition wins over the greeting.
    assert classify_intent("hi what is diabetes", SYMPTOM) == QA


@pytest.mark.parametrize(
    "query",
    [
        "are you doctor",
        "are you a doctor",
        "are you a real doctor",
        "are you human",
        "are you a bot",
        "who are you",
        "what are you",
    ],
)
def test_identity_questions_route_to_chitchat(query):
    # These must NOT produce a differential (the "are you doctor" bug).
    assert classify_intent(query, SYMPTOM) == CHITCHAT


@pytest.mark.parametrize("query", ["are you a doctor", "are you human", "who are you"])
def test_identity_reply_is_clear_about_not_being_a_doctor(query):
    reply = chitchat_reply(query)
    assert "not a human or a doctor" in reply.lower()
