// Risk Segmentation Chart
const riskCtx = document.getElementById("riskChart").getContext("2d");

new Chart(riskCtx, {
    type: "bar",
    data: {
        labels: ["Low Risk", "Medium Risk", "High Risk"],
        datasets: [{
            label: "Customer Risk Distribution",
            data: [60, 25, 15],
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: { display: false }
        }
    }
});


// Churn Probability Gauge (Line-style)
const probCtx = document.getElementById("probChart").getContext("2d");

new Chart(probCtx, {
    type: "doughnut",
    data: {
        labels: ["Churn Risk", "Safe"],
        datasets: [{
            data: [70, 30],
        }]
    },
    options: {
        responsive: true,
        cutout: "70%",
        plugins: {
            legend: { position: "bottom" }
        }
    }
});

