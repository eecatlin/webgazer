(function(window) {
    'use strict';
    
    window.webgazer = window.webgazer || {};

    const defaultWindowSize = 8;
    const equalizeStep = 5;
    const threshold = 80;
    const minCorrelation = 0.78;
    const maxCorrelation = 0.85;

    /**
     * Constructor for BlinkDetector
     * @param blinkWindow
     * @constructor
     */
    webgazer.BlinkDetector = function(blinkWindow) {
        //determines number of previous eyeObj to hold onto
        this.blinkWindow = blinkWindow || defaultWindowSize;
        this.blinkData = new webgazer.util.DataWindow(this.blinkWindow);
    };

    webgazer.BlinkDetector.prototype.extractBlinkData = function(eyesObj) {
        var useRGB = true;
        if (useRGB) {
            const eye = eyesObj.right;
            //const grayscaled = webgazer.util.grayscale(eye.patch.data, eye.width, eye.height);
            const redImage = webgazer.util.rgbExtract(eye.patch.data, 0);
            const greenImage = webgazer.util.rgbExtract(eye.patch.data, 1);
            const blueImage = webgazer.util.rgbExtract(eye.patch.data, 2);
            //const equalized = webgazer.util.equalizeHistogram(grayscaled, equalizeStep, grayscaled);
            const redEqualized = webgazer.util.equalizeHistogram(redImage, equalizeStep, redImage);
            const greenEqualized = webgazer.util.equalizeHistogram(greenImage, equalizeStep, greenImage);
            const blueEqualized = webgazer.util.equalizeHistogram(blueImage, equalizeStep, blueImage);
            //const thresholded = webgazer.util.threshold(equalized, threshold);
            const thresholdedRed = webgazer.util.threshold(redEqualized, threshold);
            const thresholdedGreen = webgazer.util.threshold(greenEqualized, threshold);
            const thresholdedBlue = webgazer.util.threshold(blueEqualized, threshold);
            const thresholdedRGB = webgazer.util.rgbConstruct(thresholdedRed, thresholdedGreen, thresholdedBlue);
            const grayscaledThreshold = webgazer.util.grayscale(thresholdedRGB, eye.width, eye.height);
        } else {
            const grayscaled = webgazer.util.grayscale(eye.patch.data, eye.width, eye.height);
            const equalized = webgazer.util.equalizeHistogram(grayscaled, equalizeStep, grayscaled);
            const grayscaledThresholded = webgazer.util.threshold(equalized, threshold);
        }
        
        return {
            data: grayscaledThreshold,
            width: eye.width,
            height: eye.height,
        };
    }

    webgazer.BlinkDetector.prototype.isSameEye = function(oldEye, newEye) {
        return (oldEye.width === newEye.width) && (oldEye.height === newEye.height);
    }

    webgazer.BlinkDetector.prototype.isBlink = function(oldEye, newEye) {
        let correlation = 0;
        for (let i = 0; i < this.blinkWindow; i++) {
            const data = this.blinkData.get(i);
            const nextData = this.blinkData.get(i + 1);
            if (!this.isSameEye(data, nextData)) {
                return false;
            }
            correlation += webgazer.util.correlation(data.data, nextData.data);
        }
        correlation /= this.blinkWindow;
        return correlation > minCorrelation && correlation < maxCorrelation;
    }

    /**
     *
     * @param eyesObj
     * @returns {*}
     */
    webgazer.BlinkDetector.prototype.detectBlink = function(eyesObj) {
        if (!eyesObj) {
            return eyesObj;
        }

        const data = this.extractBlinkData(eyesObj);
        this.blinkData.push(data);

        eyesObj.left.blink = false;
        eyesObj.right.blink = false;

        if (this.blinkData.length < this.blinkWindow) {
            return eyesObj;
        }

        if (this.isBlink()) {
            eyesObj.left.blink = true;
            eyesObj.right.blink = true;
        }

        return eyesObj;
    };

    /**
     *
     * @param value
     * @returns {webgazer.BlinkDetector}
     */
    webgazer.BlinkDetector.prototype.setBlinkWindow = function(value) {
        if (webgazer.utils.isInt(value) && value > 0) {
            this.blinkWindow = value;
        }
        return this;
    }

}(window));

