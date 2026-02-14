// FILE: static/js/interview_timer.js
(function () {
    var display = document.getElementById("timer-display");
    if (!display) {
        return;
    }

    var form = document.querySelector("form[action$='/submit']");
    var timeoutFlag = document.getElementById("auto-timeout");
    if (!form || !timeoutFlag) {
        return;
    }

    var remaining = parseInt(display.textContent || "0", 10);
    if (isNaN(remaining) || remaining <= 0) {
        return;
    }

    function paintTimer(value) {
        display.textContent = String(value);
        if (value <= 15) {
            display.classList.add("timer-danger");
        } else {
            display.classList.remove("timer-danger");
        }
    }

    paintTimer(remaining);
    var intervalId = window.setInterval(function () {
        remaining -= 1;
        paintTimer(Math.max(remaining, 0));
        if (remaining <= 0) {
            window.clearInterval(intervalId);
            timeoutFlag.value = "1";
            form.submit();
        }
    }, 1000);
})();
