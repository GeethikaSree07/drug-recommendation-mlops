async function predictDrug() {

    const age = document.getElementById("age").value;
    const sex = document.getElementById("sex").value;
    const bp = document.getElementById("bp").value;
    const cholesterol = document.getElementById("cholesterol").value;
    const na_to_k = document.getElementById("na_to_k").value;

    const result = document.getElementById("result");

    if (!age || !sex || !bp || !cholesterol || !na_to_k) {
        result.classList.remove("hidden");
        result.innerHTML = "⚠️ Please fill in all fields.";
        return;
    }

    result.classList.remove("hidden");
    result.innerHTML = "⏳ Predicting...";

    try {

        const response = await fetch("/predict", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                Age: Number(age),
                Sex: sex,
                BP: bp,
                Cholesterol: cholesterol,
                Na_to_K: Number(na_to_k)
            })
        });

        const data = await response.json();

        if (!response.ok) {
            result.innerHTML = "❌ Prediction failed.";
            return;
        }

        result.innerHTML =
            "💊 Predicted Drug Class: " +
            data.predicted_drug_class;

    } catch (error) {

        result.innerHTML =
            "❌ Could not connect to FastAPI.";

        console.error(error);
    }
}