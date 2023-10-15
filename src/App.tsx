import React, { useEffect, useState } from "react";
import Webcam from "react-webcam";
import { CustomMobileNet, load } from "./custom-mobilenet";

const App: React.FC = () => {
	const [model, setModel] = useState<CustomMobileNet>();
	const [maxPredictions, setMaxPredictions] = useState<number>(0);
	const [labels, setLabels] = useState<string[]>([]);

	const webcam = React.useRef<Webcam>(null);

	useEffect(() => {
		loadModel();
	}, []);

	useEffect(() => {
		if (!model || !webcam.current) return;
		const number = window.requestAnimationFrame(loop);

		return () => {
			window.cancelAnimationFrame(number);
		};
	}, [model]);

	const loadModel = async () => {
		const modelURL = "model/model.json";
		const metadataURL = "model/metadata.json";

		const model = await load(modelURL, metadataURL);
		const maxPredictions = model.getTotalClasses() || 0;

		setModel(model);
		setMaxPredictions(maxPredictions);
	};

	const loop = async () => {
		await predict();
		window.requestAnimationFrame(loop);
	};

	const predict = async () => {
		const canvas = webcam.current?.getCanvas();
		if (!canvas) return;

		const prediction = (await model!.predict(canvas)) as any;

		const labels = [];
		for (let i = 0; i < maxPredictions; i++) {
			const classPrediction = prediction[i].className + ": " + prediction[i].probability.toFixed(2);
			labels.push(classPrediction);
		}

		setLabels(labels);
	};

	return (
		<div>
			<h3>test 2</h3>
			<Webcam ref={webcam} videoConstraints={{ facingMode: { exact: "user" } }} />
			{labels.map((label, index) => (
				<p key={index}>{label}</p>
			))}
		</div>
	);
};

export default App;
