// OfferGuide's sole deterministic resume layout.
//
// The editor supplies all semantics, emphasis, ordering, and page decisions.
// This file only defines stable visual components and never infers meaning from
// section titles, dates, companies, or bullet text.

#set page(
  paper: "a4",
  margin: (x: 13mm, top: 10mm, bottom: 11mm),
)

#let latin-font = "Times New Roman"
#let cjk-font = "KaiTi"
#let body-font = (latin-font, cjk-font)
#let ink = rgb("#000000")

#set text(
  font: body-font,
  size: 10.6pt,
  fill: ink,
  lang: "zh",
  ligatures: false,
  fallback: true,
)
#set par(
  justify: false,
  leading: 0.65em,
  spacing: 0pt,
  linebreaks: "optimized",
)
#show link: set text(fill: ink)
#show strong: set text(
  font: body-font,
  weight: "bold",
  stroke: 0.075pt + ink,
)

#let resume-identity(name, lines: ()) = align(center)[
  #text(size: 19pt, weight: "bold", name)
  #v(1.7mm)
  #set text(size: 10.5pt)
  #if lines.len() > 0 {
    stack(
      dir: ttb,
      spacing: 2.0mm,
      ..lines.map(line => block(line)),
    )
  }
]

#let resume-header(name, lines: (), portrait: none) = block(
  width: 100%,
  below: 2.2mm,
  breakable: false,
)[
  #if portrait == none {
    v(2mm)
    resume-identity(name, lines: lines)
  } else {
    grid(
      columns: (22mm, 1fr, 22mm),
      align: top,
      [],
      align(center + horizon, resume-identity(name, lines: lines)),
      align(right + top, image(portrait, width: 22mm, height: 28mm, fit: "cover")),
    )
  }
]

#let resume-section(title, keep-with-next: false) = block(
  above: 3.0mm,
  below: 1.25mm,
  breakable: false,
  sticky: keep-with-next,
)[
  #text(size: 11.7pt, weight: "bold", title)
  #v(0.75mm)
  #line(length: 100%, stroke: 0.65pt + ink)
]

#let resume-entry(
  body,
  rows: (),
  keep-with-next: false,
  divider-before: false,
) = block(
  above: 1.8mm,
  below: 0mm,
  breakable: true,
)[
  #if rows.len() > 0 or divider-before {
    block(
      breakable: false,
      sticky: keep-with-next,
    )[
      #if divider-before {
        line(length: 100%, stroke: 0.45pt + ink)
        v(1.2mm)
      }
      #if rows.len() > 0 {
        stack(
          dir: ttb,
          spacing: 2.5mm,
          ..rows.map(row => {
            let primary = row.at(0)
            let trailing = row.at(1)
            if trailing == none {
              block(width: 100%, text(size: 10.7pt, primary))
            } else {
              grid(
                columns: (1fr, 39mm),
                column-gutter: 4mm,
                align: top,
                text(size: 10.7pt, primary),
                align(right, text(size: 10.2pt, trailing)),
              )
            }
          }),
        )
        v(2.2mm)
      }
    ]
  }
  #body
  #v(1.2mm)
]

#let resume-paragraph(body) = block(below: 1.5mm)[
  #text(size: 10.5pt, body)
]

#let resume-bullet(body) = block(below: 2.0mm)[
  #grid(
    columns: (2.8mm, 1fr),
    column-gutter: 0.45mm,
    align: (left, top),
    text(size: 8.7pt, baseline: 0pt, "•"),
    text(size: 10.5pt, body),
  )
]
