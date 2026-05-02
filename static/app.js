const form = document.querySelector("#customer-form");
const score = document.querySelector("#score");
const band = document.querySelector("#band");
const meterFill = document.querySelector("#meter-fill");
const drivers = document.querySelector("#drivers");
const actions = document.querySelector("#actions");
const accuracy = document.querySelector("#accuracy");
const rows = document.querySelector("#rows");
const baseline = document.querySelector("#baseline");
const sample = document.querySelector("#sample");

const examples = {
  risky: {
    Age: 52,
    Gender: "Female",
    Tenure: 8,
    "Usage Frequency": 4,
    "Support Calls": 8,
    "Payment Delay": 21,
    "Subscription Type": "Basic",
    "Contract Length": "Monthly",
    "Total Spend": 420,
    "Last Interaction": 24,
  },
  stable: {
    Age: 34,
    Gender: "Male",
    Tenure: 44,
    "Usage Frequency": 25,
    "Support Calls": 1,
    "Payment Delay": 2,
    "Subscription Type": "Premium",
    "Contract Length": "Annual",
    "Total Spend": 880,
    "Last Interaction": 5,
  },
};

function payloadFromForm() {
  const data = new FormData(form);
  return Object.fromEntries(data.entries());
}

function fillForm(payload) {
  for (const [key, value] of Object.entries(payload)) {
    const field = form.elements[key];
    if (field) {
      field.value = value;
      updateOutput(field);
    }
  }
}

function updateOutput(input) {
  const output = document.querySelector(`output[data-for="${input.name}"]`);
  if (output) {
    output.value = input.value;
    output.textContent = input.value;
  }
}

function renderResult(result) {
  const pct = Math.round(result.churn_probability * 100);
  score.textContent = `${pct}%`;
  meterFill.style.width = `${pct}%`;
  band.textContent = result.risk_band;
  band.className = `band ${result.risk_band}`;

  drivers.innerHTML = result.drivers
    .map((driver) => {
      const width = Math.min(100, Math.max(18, Math.abs(driver.impact) * 70));
      return `
        <article class="driver">
          <strong>${driver.feature}</strong>
          <span class="bar"><span style="width:${width}%"></span></span>
          <small>${driver.direction}</small>
        </article>
      `;
    })
    .join("");

  actions.innerHTML = result.actions.map((action) => `<li>${action}</li>`).join("");

  const model = result.model;
  accuracy.textContent = `${Math.round(model.holdout_accuracy * 100)}% holdout accuracy`;
  rows.textContent = model.training_rows.toLocaleString();
  baseline.textContent = `${Math.round(model.baseline_churn_rate * 100)}%`;
  sample.textContent = model.sample_rows.toLocaleString();
}

async function analyze() {
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payloadFromForm()),
  });
  if (!response.ok) {
    throw new Error("Prediction request failed");
  }
  renderResult(await response.json());
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  analyze().catch((error) => {
    actions.innerHTML = `<li>${error.message}</li>`;
  });
});

form.querySelectorAll('input[type="range"]').forEach((input) => {
  input.addEventListener("input", () => updateOutput(input));
});

document.querySelector("#load-risky").addEventListener("click", () => {
  fillForm(examples.risky);
  analyze();
});

document.querySelector("#load-stable").addEventListener("click", () => {
  fillForm(examples.stable);
  analyze();
});

fillForm(examples.risky);
analyze();
