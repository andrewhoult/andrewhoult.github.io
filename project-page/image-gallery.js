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
		// create big image and youtube video elements
		const mainDiv = document.createElement("div");
		const mainImg = document.createElement("img");
		const ytEmbed = document.createElement("iframe");

		// style them
		mainImg.style.display = "none";
		ytEmbed.style.display = "none";
		ytEmbed.setAttribute("frameborder", 0);
		ytEmbed.setAttribute("allow", "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share");
		ytEmbed.setAttribute("referrerpolicy", "strict-origin-when-cross-origin");
		ytEmbed.allowFullscreen = true;

		// add them to DOM
		this.appendChild(mainDiv);
		this.mainDiv = mainDiv;
		this.mainDiv.className = "gallery-display"

		mainDiv.appendChild(mainImg);
		this.mainImg = mainImg;

		mainDiv.appendChild(ytEmbed);
		this.mainVid = ytEmbed;

		// create thumbnails
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

				case "youtube":
					this.addYoutube(media.getAttribute("src"));
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

	// add an image to the gallery
	addImage(url) {
		const img = document.createElement("img");
		img.src = url;

		this.thumbnails.appendChild(img);

		if (!this.mainImg.src && !this.mainVid.src) {
			this.imgClick(img, url, "image", true);
		}

		img.onclick = () => { this.imgClick(img, url, "image") };
	}

	// add a youtube video to the gallery
	addYoutube(url) {
		const img = document.createElement("img");
		img.src = `http://i3.ytimg.com/vi/${url}/hqdefault.jpg`;

		this.thumbnails.appendChild(img);

		if (!this.mainImg.src && !this.mainVid.src) {
			this.imgClick(img, url, "youtube", true);
		}

		img.onclick = () => { this.imgClick(img, url, "youtube") };
	}

	// called when a thumbnail is clicked
	imgClick(img, url, type, dontScroll) {
		this.mainImg.style.display = "none";
		this.mainVid.style.display = "none";

		// stop youtube video if playing
		var iframeSrc = this.mainVid.src;
		this.mainVid.src = iframeSrc;

		if (type == "image") {
			this.mainImg.src = url;
			this.mainImg.style.display = "block";
		} else if (type == "youtube") {
			this.mainVid.src = `https://www.youtube.com/embed/${url}?si=VFHY-VUWjMgN1XSx`;
			this.mainVid.style.display = "block";
		}

		if (!dontScroll)
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

customElements.define("gallery-viewer", Gallery);
customElements.define("gallery-media", GalleryMedia);