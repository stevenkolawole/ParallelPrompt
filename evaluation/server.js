const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
const port = 3000;

// Serve static files (HTML, CSS, JS)
app.use(express.static(path.join(__dirname, 'public')));

// Endpoint to get the evaluations JSON
app.get('/evaluations', (req, res) => {
    const evaluationsFile = process.argv[2];
    if (!evaluationsFile) {
        return res.status(400).send('Evaluations file not specified.');
    }

    fs.readFile(evaluationsFile, 'utf8', (err, data) => {
        if (err) {
            return res.status(500).send('Error reading evaluations file.');
        }
        res.json(JSON.parse(data));
    });
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
