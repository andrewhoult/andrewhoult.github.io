let isVertical = false;

window.onresize = () => {
	const oldIsVertical = isVertical;

	const ratio = window.innerWidth / window.innerHeight;
	isVertical = ratio < 1.3;

	if (isVertical != oldIsVertical) {
		document.dispatchEvent(new Event("switchRatio"));

		if (isVertical) {
			document.getRootNode().childNodes[1].childNodes[0].innerHTML += `<link rel="stylesheet" href="mobile-style.css" id="mobile-style-override">`
		} else {
			const override = document.getElementById("mobile-style-override");

			if (override != null) {
				override.remove();
			}
		}
	}
}