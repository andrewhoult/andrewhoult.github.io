class ProjectCard {
	constructor(description, link, imageUrl) {
		this.description = description;
		this.link = link;
		this.imageUrl = imageUrl;
	}

	getHTML() {
		return `
		<a href="${this.link}">
			<div class="project-card">
				<img src="${this.imageUrl}" alt="Project thumbnail"/>
				<p>${this.description}</p>
			</div>
		</a>
		`
	}
}