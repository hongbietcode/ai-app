import * as tf from "@tensorflow/tfjs";

const IMAGE_SIZE = 224;

export interface Metadata {
	modelName?: string;
	timeStamp?: string;
	labels: string[];
	userMetadata?: {};
	grayscale?: boolean;
	imageSize?: number;
}

/**
 * Receives a Metadata object and fills in the optional fields such as timeStamp
 * @param data a Metadata object
 */
const fillMetadata = (data: Partial<Metadata>) => {
	data.timeStamp = data.timeStamp || new Date().toISOString();
	data.userMetadata = data.userMetadata || {};
	data.modelName = data.modelName || "untitled";
	data.labels = data.labels || [];
	data.imageSize = data.imageSize || IMAGE_SIZE;
	return data as Metadata;
};

// tslint:disable-next-line:no-any
const isMetadata = (c: any): c is Metadata => !!c && Array.isArray(c.labels);

/**
 * process either a URL string or a Metadata object
 * @param metadata a url to load metadata or a Metadata object
 */
const processMetadata = async (metadata: string | Metadata) => {
	let metadataJSON: Metadata;
	if (typeof metadata === "string") {
		const metadataResponse = await fetch(metadata);
		metadataJSON = await metadataResponse.json();
	} else if (isMetadata(metadata)) {
		metadataJSON = metadata;
	} else {
		throw new Error("Invalid Metadata provided");
	}
	return fillMetadata(metadataJSON);
};

export type ClassifierInputSource = HTMLImageElement | HTMLCanvasElement | HTMLVideoElement | ImageBitmap;

/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */
export async function getTopKClasses(labels: string[], logits: tf.Tensor<tf.Rank>, topK = 3) {
	const values = await logits.data();
	return tf.tidy(() => {
		topK = Math.min(topK, values.length);

		const valuesAndIndices = [];
		for (let i = 0; i < values.length; i++) {
			valuesAndIndices.push({ value: values[i], index: i });
		}
		valuesAndIndices.sort((a, b) => {
			return b.value - a.value;
		});
		const topkValues = new Float32Array(topK);
		const topkIndices = new Int32Array(topK);
		for (let i = 0; i < topK; i++) {
			topkValues[i] = valuesAndIndices[i].value;
			topkIndices[i] = valuesAndIndices[i].index;
		}

		const topClassesAndProbs = [];
		for (let i = 0; i < topkIndices.length; i++) {
			topClassesAndProbs.push({
				className: labels[topkIndices[i]],
				probability: topkValues[i],
			});
		}
		return topClassesAndProbs;
	});
}

export class CustomMobileNet {
	/**
	 * the truncated mobilenet model we will train on top of
	 */
	protected truncatedModel!: tf.LayersModel;

	static get EXPECTED_IMAGE_SIZE() {
		return IMAGE_SIZE;
	}

	protected _metadata: Metadata;
	public getMetadata() {
		return this._metadata;
	}

	constructor(public model: tf.LayersModel, metadata: Partial<Metadata>) {
		this._metadata = fillMetadata(metadata);
	}

	/**
	 * get the total number of classes existing within model
	 */
	getTotalClasses() {
		const output = this.model.output as tf.SymbolicTensor;
		const totalClasses = output.shape[1];
		return totalClasses;
	}

	/**
	 * get the model labels
	 */
	getClassLabels() {
		return this._metadata.labels;
	}

	/**
	 * Given an image element, makes a prediction through mobilenet returning the
	 * probabilities of the top K classes.
	 * @param image the image to classify
	 * @param maxPredictions the maximum number of classification predictions
	 */
	async predictTopK(image: ClassifierInputSource, maxPredictions = 10, flipped = false) {
		const croppedImage = cropTo(image, this._metadata.imageSize!, flipped);

		const logits = tf.tidy(() => {
			const captured = capture(croppedImage, this._metadata.grayscale);
			return this.model.predict(captured);
		});

		// Convert logits to probabilities and class names.
		const classes = await getTopKClasses(this._metadata.labels, logits as tf.Tensor<tf.Rank>, maxPredictions);
		tf.dispose(logits);

		return classes;
	}

	/**
	 * Given an image element, makes a prediction through mobilenet returning the
	 * probabilities for ALL classes.
	 * @param image the image to classify
	 * @param flipped whether to flip the image on X
	 */
	async predict(image: ClassifierInputSource, flipped = false) {
		const croppedImage = cropTo(image, this._metadata.imageSize!, flipped);

		const logits = tf.tidy(() => {
			const captured = capture(croppedImage, this._metadata.grayscale);
			return this.model.predict(captured);
		});

		const values = await (logits as tf.Tensor<tf.Rank>).data();

		const classes = [];
		for (let i = 0; i < values.length; i++) {
			classes.push({ className: this._metadata.labels[i], probability: values[i] });
		}

		tf.dispose(logits);

		return classes;
	}

	public dispose() {
		this.truncatedModel.dispose();
	}
}

export async function load(model: string, metadata?: string | Metadata) {
	const customModel = await tf.loadLayersModel(model);
	const metadataJSON = metadata ? await processMetadata(metadata) : null;

	// @ts-ignore
	return new CustomMobileNet(customModel, metadataJSON);
}

const newCanvas = () => document.createElement("canvas");

function cropTo(image: any, size: number, flipped = false, canvas: HTMLCanvasElement = newCanvas()) {
	// image image, bitmap, or canvas
	let width = image.width;
	let height = image.height;

	// if video element
	if (image instanceof HTMLVideoElement) {
		width = (image as HTMLVideoElement).videoWidth;
		height = (image as HTMLVideoElement).videoHeight;
	}

	const min = Math.min(width, height);
	const scale = size / min;
	const scaledW = Math.ceil(width * scale);
	const scaledH = Math.ceil(height * scale);
	const dx = scaledW - size;
	const dy = scaledH - size;
	canvas.width = canvas.height = size;
	const ctx: CanvasRenderingContext2D = canvas.getContext("2d")!;
	ctx.drawImage(image, ~~(dx / 2) * -1, ~~(dy / 2) * -1, scaledW, scaledH);

	// canvas is already sized and cropped to center correctly
	if (flipped) {
		ctx.scale(-1, 1);
		ctx.drawImage(canvas, size * -1, 0);
	}

	return canvas;
}

function capture(rasterElement: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement, grayscale?: boolean) {
	return tf.tidy(() => {
		const pixels = tf.browser.fromPixels(rasterElement);

		const cropped = cropTensor(pixels, grayscale);

		const batchedImage = cropped.expandDims(0);

		return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
	});
}

function cropTensor(img: tf.Tensor3D, grayscaleModel?: boolean, grayscaleInput?: boolean): tf.Tensor3D {
	const size = Math.min(img.shape[0], img.shape[1]);
	const centerHeight = img.shape[0] / 2;
	const beginHeight = centerHeight - size / 2;
	const centerWidth = img.shape[1] / 2;
	const beginWidth = centerWidth - size / 2;

	if (grayscaleModel && !grayscaleInput) {
		let grayscale_cropped = img.slice([beginHeight, beginWidth, 0], [size, size, 3]);

		grayscale_cropped = grayscale_cropped.reshape([size * size, 1, 3]);
		const rgb_weights = [0.2989, 0.587, 0.114];
		grayscale_cropped = tf.mul(grayscale_cropped, rgb_weights);
		grayscale_cropped = grayscale_cropped.reshape([size, size, 3]);

		grayscale_cropped = tf.sum(grayscale_cropped, -1);
		grayscale_cropped = tf.expandDims(grayscale_cropped, -1);

		return grayscale_cropped;
	}
	return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
}
