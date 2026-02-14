// FILE: static/js/interview_voice.js
(function () {
    var micToggleBtn = document.getElementById("mic-toggle");
    var statusEl = document.getElementById("voice-status");
    var answerEl = document.getElementById("answer");

    if (!micToggleBtn || !statusEl || !answerEl) {
        return;
    }

    var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        statusEl.textContent = "Voice input not supported in this browser.";
        micToggleBtn.disabled = true;
        return;
    }

    var recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.continuous = true;
    recognition.interimResults = true;

    var finalTranscript = "";
    var baseText = "";
    var isRunning = false;

    function setRunningState(running) {
        isRunning = running;
        micToggleBtn.classList.toggle("recording", running);
        micToggleBtn.setAttribute("aria-label", running ? "Stop recording" : "Start recording");
        micToggleBtn.title = running ? "Stop recording" : "Start recording";
        statusEl.textContent = running ? "Listening..." : "Idle";
    }

    recognition.onstart = function () {
        baseText = answerEl.value.trim();
        finalTranscript = "";
        setRunningState(true);
    };

    recognition.onresult = function (event) {
        var interimTranscript = "";
        for (var i = event.resultIndex; i < event.results.length; i += 1) {
            var chunk = event.results[i][0].transcript || "";
            if (event.results[i].isFinal) {
                finalTranscript += chunk.trim() + " ";
            } else {
                interimTranscript += chunk;
            }
        }

        var speechText = (finalTranscript + interimTranscript).trim();
        var chunks = [];
        if (baseText) {
            chunks.push(baseText);
        }
        if (speechText) {
            chunks.push(speechText);
        }
        answerEl.value = chunks.join(" ").trim();
    };

    recognition.onerror = function (event) {
        statusEl.textContent = "Voice error: " + event.error;
        setRunningState(false);
    };

    recognition.onend = function () {
        setRunningState(false);
        finalTranscript = "";
        baseText = answerEl.value.trim();
    };

    micToggleBtn.addEventListener("click", function () {
        if (isRunning) {
            recognition.stop();
            return;
        }
        try {
            recognition.start();
        } catch (err) {
            statusEl.textContent = "Unable to start voice input.";
            setRunningState(false);
        }
    });
})();
