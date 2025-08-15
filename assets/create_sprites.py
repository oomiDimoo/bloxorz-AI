from PIL import Image, ImageDraw

def create_sprite(filename, color, size=(64, 64)):
    """Creates a simple sprite with a border and saves it."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw a filled rectangle with a border
    draw.rectangle([0, 0, size[0]-1, size[1]-1], fill=color, outline="black", width=2)

    img.save(f"assets/images/{filename}", "PNG")

if __name__ == "__main__":
    # Create floor tile
    create_sprite("floor.png", (200, 200, 200))

    # Create block face
    create_sprite("block_face.png", (255, 100, 100))

    # Create goal tile
    create_sprite("goal.png", (100, 255, 100))

    print("Sprites created successfully.")
