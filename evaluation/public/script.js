let evaluations = [];
let currentEvaluationIndex = 0;
let results = [];
const questions = [
    "Which output more accurately answers the prompt?",
    "Which output is more grammatically correct?",
    "Which output is more detailed?",
    "Which output do you prefer overall?"
];

document.addEventListener('DOMContentLoaded', async () => {
    evaluations = await fetchEvaluations();
    evaluations = shuffle(evaluations);
    renderEvaluation();
    document.getElementById('save-results').addEventListener('click', saveResults);
    document.addEventListener('keydown', handleKeydown);
});

async function fetchEvaluations() {
    const response = await fetch('/evaluations');
    if (!response.ok) {
        alert('Failed to load evaluations');
        return [];
    }
    return await response.json();
}

function renderEvaluation() {
    const evaluationContainer = document.getElementById('evaluation-container');
    evaluationContainer.innerHTML = '';

    const currentEvaluation = evaluations[currentEvaluationIndex];
    const promptContainer = document.createElement('div');
    promptContainer.classList.add('prompt-container');
    promptContainer.innerHTML = `<label>Prompt:</label><textarea readonly>${currentEvaluation.prompt}</textarea>`;

    // Randomize the order of outputs and track which is which
    const outputs = shuffle([
        { type: 'serial', content: currentEvaluation.serial_output },
        { type: 'parallel', content: currentEvaluation.parallel_output }
    ]);

    const outputContainer = document.createElement('div');
    outputContainer.classList.add('output-container');
    outputContainer.innerHTML = `
        <label>Output 1:</label><textarea readonly>${outputs[0].content}</textarea>
        <label>Output 2:</label><textarea readonly>${outputs[1].content}</textarea>
    `;

    const questionsContainer = document.createElement('div');
    questionsContainer.classList.add('questions-container');

    questions.forEach((question, index) => {
        const questionDiv = document.createElement('div');
        questionDiv.classList.add('question');
        questionDiv.innerHTML = `
            <label>${question}</label>
            <div class="radio-group">
                <label><input type="radio" name="question${index}" value="${outputs[0].type}"> Output 1</label>
                <label><input type="radio" name="question${index}" value="${outputs[1].type}"> Output 2</label>
                <label><input type="radio" name="question${index}" value="tie"> Tie</label>
            </div>
        `;
        questionsContainer.appendChild(questionDiv);
    });

    evaluationContainer.appendChild(promptContainer);
    evaluationContainer.appendChild(outputContainer);
    evaluationContainer.appendChild(questionsContainer);
}

function handleKeydown(event) {
    if (event.key === 'ArrowRight') {
        saveCurrentEvaluation();
        if (currentEvaluationIndex < evaluations.length - 1) {
            currentEvaluationIndex++;
            renderEvaluation();
        }
    }
    if (event.key === 'ArrowLeft') {
        if (currentEvaluationIndex > 0) {
            currentEvaluationIndex--;
            renderEvaluation();
        }
    }
}

function saveCurrentEvaluation() {
    const currentEvaluation = evaluations[currentEvaluationIndex];
    const answers = questions.map((_, index) => {
        const selectedOption = document.querySelector(`input[name="question${index}"]:checked`);
        return selectedOption ? selectedOption.value : null;
    });

    results.push({
        id: currentEvaluation.id,
        prompt: currentEvaluation.prompt,
        answers
    });
}

function saveResults() {
    saveCurrentEvaluation();
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'evaluation_results.json';
    a.click();
}

function shuffle(array) {
    let currentIndex = array.length, randomIndex;

    while (currentIndex !== 0) {
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;
        [array[currentIndex], array[randomIndex]] = [array[randomIndex], array[currentIndex]];
    }

    return array;
}
