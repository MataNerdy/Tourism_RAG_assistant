import AppKit

let root = URL(fileURLWithPath: "/Users/Di/Documents/GitHub/Tourism_RAG_assistant")
let images = root.appendingPathComponent("images")
let sourceImages = URL(fileURLWithPath: "/Users/Di/Documents/GitHub/DeepLearning/Tourism_RAG_assistant/assets")

struct HistogramBin: Decodable {
    let start: Double
    let end: Double
    let count: Double
}

struct DistributionData: Decodable {
    let raw_valid_rows: Int
    let after_caption_dedup: Int
    let unique_wikidata: Int
    let unique_place_hints: Int
    let duplicates_total: Int
    let max_duplicates: Double
    let mean_duplicates: Double
    let median_duplicates: Double
    let histogram: [HistogramBin]
}

func attrs(size: CGFloat, weight: NSFont.Weight = .regular, color: NSColor = .black) -> [NSAttributedString.Key: Any] {
    [
        .font: NSFont.systemFont(ofSize: size, weight: weight),
        .foregroundColor: color
    ]
}

func drawText(_ text: String, x: CGFloat, yTop: CGFloat, size: CGFloat, weight: NSFont.Weight = .regular, color: NSColor = .black) {
    let attributed = NSAttributedString(string: text, attributes: attrs(size: size, weight: weight, color: color))
    attributed.draw(at: NSPoint(x: x, y: yTop))
}

func drawCentered(_ text: String, centerX: CGFloat, yTop: CGFloat, size: CGFloat, weight: NSFont.Weight = .regular, color: NSColor = .black) {
    let attributed = NSAttributedString(string: text, attributes: attrs(size: size, weight: weight, color: color))
    let textSize = attributed.size()
    attributed.draw(at: NSPoint(x: centerX - textSize.width / 2.0, y: yTop))
}

func drawRotatedCentered(_ text: String, centerX: CGFloat, centerY: CGFloat, size: CGFloat) {
    let attributed = NSAttributedString(string: text, attributes: attrs(size: size))
    let textSize = attributed.size()
    let transform = NSAffineTransform()
    transform.translateX(by: centerX, yBy: centerY)
    transform.rotate(byDegrees: 90)
    transform.concat()
    attributed.draw(at: NSPoint(x: -textSize.width / 2.0, y: -textSize.height / 2.0))
    transform.invert()
    transform.concat()
}

func savePNG(name: String, width: CGFloat, height: CGFloat, draw: () -> Void) {
    let image = NSImage(size: NSSize(width: width, height: height))
    image.lockFocus()
    NSColor.white.setFill()
    NSRect(x: 0, y: 0, width: width, height: height).fill()
    draw()
    image.unlockFocus()

    guard let tiff = image.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: tiff),
          let png = bitmap.representation(using: .png, properties: [:]) else {
        fatalError("Could not render \(name)")
    }
    try! png.write(to: images.appendingPathComponent(name))
}

func redrawDataDistribution() {
    let dataURL = images.appendingPathComponent("data_distribution_data.json")
    let data = try! Data(contentsOf: dataURL)
    let distribution = try! JSONDecoder().decode(DistributionData.self, from: data)
    let width: CGFloat = 980
    let height: CGFloat = 560

    savePNG(name: "data_distribution.png", width: width, height: height) {
        let plot = NSRect(x: 88, y: 88, width: 805, height: 360)
        let maxY: CGFloat = 80
        let xMax: CGFloat = 125

        NSColor(calibratedRed: 0.98, green: 0.99, blue: 1.0, alpha: 1).setFill()
        NSRect(x: 0, y: 0, width: width, height: height).fill()

        drawCentered("Distribution of records per place", centerX: width / 2, yTop: height - 44, size: 22, weight: .semibold)
        drawCentered("Duplicate-heavy landmarks are consolidated before building the retrieval corpus", centerX: width / 2, yTop: height - 70, size: 13, color: NSColor(calibratedWhite: 0.35, alpha: 1))

        NSColor.white.setFill()
        plot.fill()

        for tick in stride(from: 0, through: 80, by: 10) {
            let y = plot.minY + CGFloat(tick) / maxY * plot.height
            NSColor(calibratedWhite: 0.84, alpha: 1).setStroke()
            let grid = NSBezierPath()
            grid.move(to: NSPoint(x: plot.minX, y: y))
            grid.line(to: NSPoint(x: plot.maxX, y: y))
            grid.setLineDash([5, 4], count: 2, phase: 0)
            grid.lineWidth = 1
            grid.stroke()
            drawText(String(tick), x: 50, yTop: y - 7, size: 11, color: NSColor(calibratedWhite: 0.25, alpha: 1))
        }

        NSColor(calibratedRed: 0.40, green: 0.63, blue: 0.88, alpha: 0.92).setFill()
        for bin in distribution.histogram {
            let x0 = plot.minX + CGFloat(bin.start) / xMax * plot.width
            let x1 = plot.minX + CGFloat(bin.end) / xMax * plot.width
            let barHeight = CGFloat(bin.count) / maxY * plot.height
            NSRect(x: x0 + 1, y: plot.minY, width: max(1, x1 - x0 - 2), height: barHeight).fill()
        }

        NSColor(calibratedWhite: 0.12, alpha: 1).setStroke()
        let axes = NSBezierPath()
        axes.move(to: NSPoint(x: plot.minX, y: plot.minY))
        axes.line(to: NSPoint(x: plot.maxX, y: plot.minY))
        axes.move(to: NSPoint(x: plot.minX, y: plot.minY))
        axes.line(to: NSPoint(x: plot.minX, y: plot.maxY))
        axes.lineWidth = 1.5
        axes.stroke()

        for tick in stride(from: 0, through: 120, by: 20) {
            let x = plot.minX + CGFloat(tick) / xMax * plot.width
            drawCentered(String(tick), centerX: x, yTop: plot.minY - 22, size: 11, color: NSColor(calibratedWhite: 0.25, alpha: 1))
        }

        let medianX = plot.minX + CGFloat(distribution.median_duplicates) / xMax * plot.width
        NSColor.systemOrange.setStroke()
        let medianLine = NSBezierPath()
        medianLine.move(to: NSPoint(x: medianX, y: plot.minY))
        medianLine.line(to: NSPoint(x: medianX, y: plot.maxY))
        medianLine.setLineDash([2, 4], count: 2, phase: 0)
        medianLine.lineWidth = 2.5
        medianLine.stroke()

        drawRotatedCentered("Number of places", centerX: 24, centerY: plot.midY, size: 13)
        drawCentered("Number of duplicate records per place", centerX: plot.midX, yTop: 48, size: 13)

        NSColor.systemOrange.setStroke()
        let line = NSBezierPath()
        line.move(to: NSPoint(x: width - 165, y: height - 101))
        line.line(to: NSPoint(x: width - 130, y: height - 101))
        line.setLineDash([2, 3], count: 2, phase: 0)
        line.lineWidth = 2
        line.stroke()
        drawText("median = \(Int(distribution.median_duplicates))", x: width - 122, yTop: height - 109, size: 13)

        let summary = "Original notebook preprocessing: \(distribution.raw_valid_rows) valid rows -> \(distribution.after_caption_dedup) caption-deduplicated rows -> \(distribution.unique_place_hints) unique places"
        drawCentered(summary, centerX: width / 2, yTop: 24, size: 11, color: NSColor(calibratedWhite: 0.34, alpha: 1))
    }
}

func redrawProjection(name: String, title: String) {
    let source = NSImage(contentsOf: sourceImages.appendingPathComponent("\(name).png"))!
    let width = source.size.width
    let height = source.size.height
    let labels: [(NSColor, String)] = [
        (NSColor(calibratedRed: 0.388, green: 0.431, blue: 0.980, alpha: 1), "Vladimir"),
        (NSColor(calibratedRed: 0.937, green: 0.325, blue: 0.231, alpha: 1), "Yekaterinburg"),
        (NSColor(calibratedRed: 0.000, green: 0.800, blue: 0.588, alpha: 1), "Nizhny Novgorod"),
        (NSColor(calibratedRed: 0.671, green: 0.388, blue: 0.980, alpha: 1), "Yaroslavl"),
    ]

    savePNG(name: "\(name).png", width: width, height: height) {
        source.draw(in: NSRect(x: 0, y: 0, width: width, height: height))
        NSColor.white.setFill()
        NSRect(x: 95, y: height - 128, width: 930, height: 86).fill()
        NSRect(x: width - 290, y: height - 390, width: 290, height: 260).fill()

        drawText(title, x: 145, yTop: height - 100, size: 28, weight: .medium, color: NSColor(calibratedWhite: 0.23, alpha: 1))
        drawText("city", x: width - 255, yTop: height - 205, size: 22, color: NSColor(calibratedWhite: 0.23, alpha: 1))
        for (index, item) in labels.enumerated() {
            let y = height - CGFloat(226 + index * 32)
            item.0.setFill()
            NSBezierPath(ovalIn: NSRect(x: width - 248, y: y, width: 10, height: 10)).fill()
            drawText(item.1, x: width - 218, yTop: y - 8, size: 22, color: NSColor(calibratedWhite: 0.23, alpha: 1))
        }
    }
}

func roundedRect(_ rect: NSRect, radius: CGFloat, fill: NSColor, stroke: NSColor? = nil, lineWidth: CGFloat = 1) {
    let path = NSBezierPath(roundedRect: rect, xRadius: radius, yRadius: radius)
    fill.setFill()
    path.fill()
    if let stroke = stroke {
        stroke.setStroke()
        path.lineWidth = lineWidth
        path.stroke()
    }
}

func redrawPipeline(name: String) {
    let width: CGFloat = 1800
    let height: CGFloat = 760
    let cols: [(String, String, String, [String], NSColor)] = [
        ("1", "Data Ingestion", "Raw sources", ["WikiData metadata", "BLIP image captions", "Coordinates and city labels", "CSV input rows"], .systemGreen),
        ("2", "Cleaning", "Corpus preparation", ["Normalize text fields", "Drop exact caption duplicates", "Regex noise filtering", "MiniLM semantic consistency"], .systemBlue),
        ("3", "Aggregation", "One place, one document", ["Group by WikiData", "Pick representative captions", "Merge metadata + descriptions", "Sample 250 final docs"], .systemPurple),
        ("4", "Vector Store", "Dense retrieval", ["multilingual-e5-base", "ChromaDB persistence", "Top-k candidate search", "Deduplicate passages"], .systemOrange),
        ("5", "Answering", "Rerank and generate", ["ColBERTv2 reranking", "Top context passages", "Grounded prompt template", "Mistral-7B-Instruct"], .systemTeal),
    ]

    savePNG(name: name, width: width, height: height) {
        drawCentered("Tourism RAG Assistant Architecture", centerX: width / 2, yTop: height - 62, size: 42, weight: .bold, color: NSColor(calibratedRed: 0.06, green: 0.09, blue: 0.16, alpha: 1))
        drawCentered("From noisy tourism records to retrieved context and grounded answers", centerX: width / 2, yTop: height - 100, size: 22, color: .darkGray)

        let cardW: CGFloat = 300
        let cardH: CGFloat = 410
        let gap: CGFloat = 40
        let startX: CGFloat = 40
        let y: CGFloat = 220

        for (i, col) in cols.enumerated() {
            let x = startX + CGFloat(i) * (cardW + gap)
            roundedRect(NSRect(x: x, y: y, width: cardW, height: cardH), radius: 16, fill: .white, stroke: col.4, lineWidth: 2)
            col.4.setFill()
            NSBezierPath(ovalIn: NSRect(x: x + 16, y: y + cardH - 64, width: 44, height: 44)).fill()
            drawCentered(col.0, centerX: x + 38, yTop: y + cardH - 53, size: 20, weight: .bold, color: .white)
            drawText(col.1, x: x + 72, yTop: y + cardH - 42, size: 24, weight: .bold, color: NSColor(calibratedWhite: 0.08, alpha: 1))
            drawText(col.2, x: x + 72, yTop: y + cardH - 70, size: 18, color: .darkGray)
            roundedRect(NSRect(x: x + 18, y: y + 28, width: cardW - 36, height: cardH - 130), radius: 12, fill: NSColor(calibratedRed: 0.97, green: 0.98, blue: 1.0, alpha: 1), stroke: NSColor(calibratedRed: 0.86, green: 0.89, blue: 0.94, alpha: 1), lineWidth: 1)
            for (j, bullet) in col.3.enumerated() {
                let by = y + cardH - 185 - CGFloat(j) * 48
                col.4.setFill()
                NSBezierPath(ovalIn: NSRect(x: x + 35, y: by + 5, width: 12, height: 12)).fill()
                drawText(bullet, x: x + 58, yTop: by, size: 17, color: NSColor(calibratedWhite: 0.13, alpha: 1))
            }

            if i < cols.count - 1 {
                drawCentered("→", centerX: x + cardW + gap / 2, yTop: y + cardH / 2 - 16, size: 42, weight: .bold, color: NSColor(calibratedWhite: 0.08, alpha: 1))
            }
        }

        roundedRect(NSRect(x: 110, y: 104, width: 1580, height: 76), radius: 14, fill: NSColor(calibratedRed: 0.97, green: 0.98, blue: 1.0, alpha: 1), stroke: NSColor(calibratedRed: 0.78, green: 0.83, blue: 0.89, alpha: 1), lineWidth: 1)
        let flow = ["User question", "Dense retrieval", "ColBERT rerank", "Prompt assembly", "Grounded answer"]
        for (i, step) in flow.enumerated() {
            let x = CGFloat(230 + i * 285)
            drawCentered(step, centerX: x, yTop: 134, size: 21, color: NSColor(calibratedWhite: 0.08, alpha: 1))
            if i < flow.count - 1 {
                drawCentered("→", centerX: x + 140, yTop: 128, size: 30, color: .darkGray)
            }
        }

        roundedRect(NSRect(x: 260, y: 28, width: 1280, height: 42), radius: 10, fill: NSColor(calibratedRed: 0.94, green: 0.97, blue: 1.0, alpha: 1), stroke: NSColor(calibratedRed: 0.75, green: 0.86, blue: 0.98, alpha: 1), lineWidth: 1)
        drawCentered("Observed corpus: 12,078 raw rows -> 8,137 cleaned rows -> 295 unique objects -> 250 final retrieval documents", centerX: width / 2, yTop: 40, size: 19, color: NSColor(calibratedRed: 0.12, green: 0.23, blue: 0.54, alpha: 1))
    }
}

redrawDataDistribution()
redrawPipeline(name: "pipeline.png")
redrawPipeline(name: "pipeline_rag.png")
