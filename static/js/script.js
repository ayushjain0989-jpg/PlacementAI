/*
    PlacementAI — client-side behaviour.

    Two jobs, and deliberately no more:

      1. Draw the three analytics charts from /analytics/data.json.
      2. Add a confirm() step to forms carrying data-confirm.

    Everything this file used to be responsible for is gone. The SHAP bar widths were
    previously set here (and in an inline <script> in prediction_details.html) purely to
    work around a duplicate `.shap-bar { width: 0 }` rule in style.css. That rule has been
    deleted, so the --bar-width custom property in the markup now works on its own and the
    bars render with CSS off... well, with JavaScript off. That is the point: nothing on
    this site depends on this file to be usable.
*/

(function () {

    "use strict";


    /* =====================================================
       CONFIRM STEP FOR DESTRUCTIVE FORMS
    ===================================================== */
    /*
        Progressive enhancement, not the safety mechanism. The real guards are on the
        server: the routes are POST-only and CSRF-checked, and the templates already wrap
        each destructive action in a <details> disclosure or a type-DELETE field. This
        just catches the misclick.
    */

    function wireConfirmations() {

        var forms = document.querySelectorAll("form[data-confirm]");

        Array.prototype.forEach.call(forms, function (form) {

            form.addEventListener("submit", function (event) {

                var message = form.getAttribute("data-confirm");

                if (!window.confirm(message)) {
                    event.preventDefault();
                }

            });

        });

    }


    /* =====================================================
       CHARTS
    ===================================================== */

    /*
        One palette, used by every chart, so a readiness level is the same colour in the
        doughnut as its badge is in the tables.
    */
    var COLOURS = {
        high: "#16a34a",
        moderate: "#f59e0b",
        low: "#dc2626",
        accent: "#4f46e5",
        accentSoft: "rgba(79, 70, 229, 0.15)",
        grid: "rgba(148, 163, 184, 0.25)",
        text: "#475569"
    };


    var READINESS_COLOURS = [
        COLOURS.high,
        COLOURS.moderate,
        COLOURS.low
    ];


    /* Shared chart options. Charts sit in fixed-height holders, so they must be allowed
       to ignore their intrinsic aspect ratio and fill the box instead. */
    function baseOptions(extra) {

        var options = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: COLOURS.text,
                        font: { size: 12 }
                    }
                }
            }
        };

        return Object.assign(options, extra || {});

    }


    /* Counts are whole assessments, so the y axis must not offer 0.5 of one. */
    var COUNT_AXIS = {
        beginAtZero: true,
        ticks: {
            precision: 0,
            color: COLOURS.text
        },
        grid: {
            color: COLOURS.grid
        }
    };


    var LABEL_AXIS = {
        ticks: {
            color: COLOURS.text
        },
        grid: {
            display: false
        }
    };


    function drawReadiness(canvas, data) {

        return new Chart(canvas, {

            type: "doughnut",

            data: {
                labels: data.labels,
                datasets: [{
                    data: data.counts,
                    backgroundColor: READINESS_COLOURS,
                    borderColor: "#ffffff",
                    borderWidth: 2
                }]
            },

            options: baseOptions({
                cutout: "62%",
                plugins: {
                    legend: {
                        position: "bottom",
                        labels: {
                            color: COLOURS.text,
                            padding: 16,
                            usePointStyle: true
                        }
                    }
                }
            })

        });

    }


    function drawHistogram(canvas, data) {

        /*
            Colour by band rather than one flat colour: the bars mean "how many students
            scored 0-9, 10-19, ..." and tinting the low bands red and the high bands green
            says which end of the chart is the good end without a second legend.
        */
        var colours = data.labels.map(function (label) {

            var lower = parseInt(label, 10);

            if (isNaN(lower)) {
                return COLOURS.accent;
            }

            if (lower >= 75) {
                return COLOURS.high;
            }

            if (lower >= 50) {
                return COLOURS.moderate;
            }

            return COLOURS.low;

        });

        return new Chart(canvas, {

            type: "bar",

            data: {
                labels: data.labels,
                datasets: [{
                    label: "Assessments",
                    data: data.counts,
                    backgroundColor: colours,
                    borderRadius: 6,
                    maxBarThickness: 48
                }]
            },

            options: baseOptions({
                scales: {
                    x: LABEL_AXIS,
                    y: COUNT_AXIS
                },
                plugins: {
                    legend: { display: false }
                }
            })

        });

    }


    function drawTimeline(canvas, data) {

        return new Chart(canvas, {

            type: "line",

            data: {
                labels: data.labels,
                datasets: [{
                    label: "Assessments",
                    data: data.counts,
                    borderColor: COLOURS.accent,
                    backgroundColor: COLOURS.accentSoft,
                    borderWidth: 2,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                    pointBackgroundColor: COLOURS.accent
                }]
            },

            options: baseOptions({
                scales: {
                    x: LABEL_AXIS,
                    y: COUNT_AXIS
                },
                plugins: {
                    legend: { display: false }
                }
            })

        });

    }


    /*
        Replaces a chart's holder with a short message. Used when the fetch fails or the
        endpoint reports no data -- an empty <canvas> looks like a rendering bug, and a
        sentence does not.
    */
    function replaceWithNote(canvas, message) {

        if (!canvas) {
            return;
        }

        var note = document.createElement("p");

        note.className = "empty-note";
        note.textContent = message;

        canvas.replaceWith(note);

    }


    function drawCharts(container) {

        var readiness = document.getElementById("readiness-chart");
        var histogram = document.getElementById("histogram-chart");
        var timeline = document.getElementById("timeline-chart");

        /*
            The CDN can be blocked or offline. Without this the page would throw a
            ReferenceError and the console would fill with noise on a page that is
            otherwise perfectly readable.
        */
        if (typeof Chart === "undefined") {

            [readiness, histogram, timeline].forEach(function (canvas) {
                replaceWithNote(canvas, "Charts could not be loaded.");
            });

            return;

        }

        var url = container.getAttribute("data-analytics-url");

        if (!url) {
            return;
        }

        fetch(url, { headers: { "Accept": "application/json" } })

            .then(function (response) {

                if (!response.ok) {
                    throw new Error("Request failed: " + response.status);
                }

                return response.json();

            })

            .then(function (data) {

                if (!data || !data.total) {

                    [readiness, histogram, timeline].forEach(function (canvas) {
                        replaceWithNote(canvas, "No assessment data yet.");
                    });

                    return;

                }

                if (readiness && data.readiness) {
                    drawReadiness(readiness, data.readiness);
                }

                if (histogram && data.histogram) {
                    drawHistogram(histogram, data.histogram);
                }

                if (timeline && data.over_time) {
                    drawTimeline(timeline, data.over_time);
                }

            })

            .catch(function (error) {

                [readiness, histogram, timeline].forEach(function (canvas) {
                    replaceWithNote(canvas, "Chart data could not be loaded.");
                });

                console.error("Analytics charts:", error);

            });

    }


    /* =====================================================
       START
    ===================================================== */

    function init() {

        wireConfirmations();

        var chartGrid = document.querySelector("[data-analytics-url]");

        if (chartGrid) {
            drawCharts(chartGrid);
        }

    }


    /* This file is loaded with `defer`, so the DOM is already parsed by the time it runs;
       the readyState check covers the case where it is ever loaded some other way. */
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }

}());
