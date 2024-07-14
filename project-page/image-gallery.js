class Gallery extends HTMLElement {
	constructor() {
		super();
	}

	static observedAttributes = ["type", "src", "thumbnail", "alt"];

	updateRendering() {
		// Left as an exercise for the reader. But, you'll probably want to
		// check this.ownerDocument.defaultView to see if we've been
		// inserted into a document with a browsing context, and avoid
		// doing any work if not.
	}

	connectedCallback() {
		const mainImg = document.createElement("img");
		const mainVid = document.createElement("video");

		mainVid.style.display = "none";

		this.appendChild(mainImg);
		this.mainImg = mainImg;

		this.appendChild(mainVid);
		this.mainVid = mainVid;

		const thumbnails = document.createElement("div");
		thumbnails.className = "gallery-thumbnails";

		this.appendChild(thumbnails);
		this.thumbnails = thumbnails;

		this.mediaArray = new Array();

		const arr = this.getElementsByTagName("gallery-media");
		for (let i = 0; i < arr.length; ++i) {
			const media = arr.item(i);

			switch (media.getAttribute("type")) {
				case "image":
					this.addImage(media.getAttribute("src"));
					break;

				default:
					console.error("Unknown media type");
					continue;
			}
		}
	}

	disconnectedCallback() {

	}

	attributeChangedCallback(name, oldValue, newValue) {
		console.log(
			`Attribute ${name} has changed from ${oldValue} to ${newValue}.`,
		);
		console.error("Attribute updated not implemented yet!");
	}

	addImage(url) {
		const img = document.createElement("img");
		img.src = url;

		this.thumbnails.appendChild(img);

		if (!this.mainImg.src) {
			this.imgClick(img);
		}

		img.onclick = () => { this.imgClick(img) };
	}

	imgClick(img) {
		this.mainImg.src = img.src;
		img.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });

		if (this.selectedImage) {
			this.selectedImage.className = "";
		}

		this.selectedImage = img;
		img.className = "selected";
	}
}

class GalleryMedia extends HTMLElement {

}

document.addEventListener("DOMContentLoaded", () => {
	customElements.define("gallery-viewer", Gallery);
	customElements.define("gallery-media", GalleryMedia);
});