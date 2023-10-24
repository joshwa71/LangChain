from flask import Flask, render_template, request, jsonify
from ice_breaker import icebreak

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")



@app.route("/process", methods=["POST"])
def process():
    name = request.form["name"]
    person_intel = icebreak(name=name)
    return jsonify(
        {
            "summary": person_intel.summary,
            "facts": person_intel.facts,
            "topics_of_interest": person_intel.topics_of_interest,
            "ice_breakers": person_intel.ice_breakers
        }
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0" , debug=True)