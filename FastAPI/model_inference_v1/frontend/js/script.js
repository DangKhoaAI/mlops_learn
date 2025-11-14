document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("wine-form");
    const resultEl=document.getElementById("prediction-result");
    const submitButton = document.getElementById("submit-button");

    //api backend
    const apiUrl = "http://127.0.0.1:8000/predict"; 

    form.addEventListener("submit", async function(e) {
        e.preventDefault();

        //edit button UI
        submitButton.disabled = true;
        submitButton.textContent = "Processing...";
        resultEl.textContent = "";

        //gather form data
        const formData = new FormData(form);
        const data = {};

        //from-> js
        formData.forEach((value, key) => {
            data[key] = parseFloat(value); //float because model expect numerical input
        });

        //send request to backend
        try {
            const response = await fetch(apiUrl, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data) //js->json
            });
            
            //handle response
            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }
            //get json from response
            const result = await response.json();

            //round to get class
            const predictedClass = Math.round(result.quality);
            
            resultEl.textContent = predictedClass;
            }
        //catch errors
        catch (error) {
            console.error("Error during inference:", error);
            resultEl.textContent = "Error during inference. Please try again.";
        }
        //reset button UI
        finally {
            submitButton.disabled = false;
            submitButton.textContent = "Submit";
        }
    });
});