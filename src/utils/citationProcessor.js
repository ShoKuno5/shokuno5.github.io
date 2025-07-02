import { Cite } from '@citation-js/core'
import '@citation-js/plugin-bibtex'
import '@citation-js/plugin-csl'

class CitationProcessor {
  constructor() {
    this.bibliography = null
    this.citations = new Map()
    this.usedCitations = new Set()
  }

  // Load bibliography from .bib file content
  async loadBibliography(bibContent) {
    try {
      this.bibliography = new Cite(bibContent)
      
      // Create a map for quick lookup by citation key
      this.citations.clear()
      const data = this.bibliography.data
      
      for (let item of data) {
        if (item.id) {
          this.citations.set(item.id, item)
        }
      }
      
      return true
    } catch (error) {
      console.error('Error loading bibliography:', error)
      return false
    }
  }

  // Process content and replace \cite{key} with numbered citations
  processCitations(content) {
    this.usedCitations.clear()
    
    // Find all citations in the content
    const citeRegex = /\\cite\{([^}]+)\}/g
    let match
    const foundCitations = []
    
    while ((match = citeRegex.exec(content)) !== null) {
      const key = match[1]
      if (this.citations.has(key)) {
        foundCitations.push(key)
        this.usedCitations.add(key)
      }
    }
    
    // Sort citations alphabetically for consistent numbering
    const sortedCitations = Array.from(this.usedCitations).sort()
    
    // Replace citations with numbered links
    let processedContent = content
    sortedCitations.forEach((key, index) => {
      const citationNumber = index + 1
      const regex = new RegExp(`\\\\cite\\{${key}\\}`, 'g')
      processedContent = processedContent.replace(
        regex, 
        `<a href="#ref-${key}" class="citation-link">[${citationNumber}]</a>`
      )
    })
    
    return processedContent
  }

  // Generate bibliography HTML using Citation.js
  generateBibliography() {
    if (!this.bibliography || this.usedCitations.size === 0) {
      return ''
    }

    // Sort citations alphabetically for consistent numbering
    const sortedCitations = Array.from(this.usedCitations).sort()
    
    // Filter bibliography to only used citations
    const usedItems = sortedCitations
      .map(key => this.citations.get(key))
      .filter(item => item !== undefined)
    
    if (usedItems.length === 0) {
      return ''
    }

    // Create a new Cite instance with only used citations
    const usedBibliography = new Cite(usedItems)
    
    // Generate APA style bibliography
    const bibliographyHtml = usedBibliography.format('bibliography', {
      format: 'html',
      template: 'apa',
      lang: 'en-US'
    })
    
    // Process the generated HTML to add numbers and proper styling
    let processedHtml = bibliographyHtml
    
    // Add citation numbers to each entry
    sortedCitations.forEach((key, index) => {
      const citationNumber = index + 1
      // Find the bibliography entry and add number
      const entryPattern = new RegExp(`<div class="csl-entry"[^>]*>`, 'g')
      let entryIndex = 0
      
      processedHtml = processedHtml.replace(entryPattern, (match) => {
        if (entryIndex === index) {
          return match.replace('>', ` id="ref-${key}">
            <div class="citation-layout">
              <span class="citation-number">[${citationNumber}]</span>
              <div class="citation-content">`)
        }
        entryIndex++
        return match
      })
    })
    
    // Close the citation-content and citation-layout divs
    processedHtml = processedHtml.replace(/<\/div>(?=\s*<div class="csl-entry"|$)/g, '</div></div></div>')
    
    // Wrap in bibliography container
    const finalHtml = `
      <div class="bibliography">
        <h2 class="bibliography-title" style="color: #1f2937 !important;">References</h2>
        <div class="bibliography-list">
          ${processedHtml}
        </div>
      </div>
    `
    
    return finalHtml
  }

  // Get citation data for a specific key
  getCitation(key) {
    return this.citations.get(key)
  }

  // Get all used citations
  getUsedCitations() {
    return Array.from(this.usedCitations).sort()
  }
}

export default CitationProcessor