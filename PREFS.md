# User Preferences for Proof Writing and AI Assistant Behavior

## Mathematical Proof Style Preferences

### Completeness Requirements
- User expects rigorous, complete algebraic proofs without gaps
- When starting a proof attempt, must follow through to completion
- If gaps remain, explicitly acknowledge them rather than hand-waving
- Phrases like "it can be shown", "after extensive algebraic manipulation", "the proof proceeds with", "simplifying yields", "the pattern continues", "this case follows from the previous", "the terms combine to give the result", etc. all count as gaps and should not be permitted
- After each proof attempt, if gaps remain, go back and attempt to fill the gaps; if any remain, acknowledge them, then rinse and repeat

### Proof Structure
- Use clear section headings and subsections for organization
- Break complex proofs into manageable steps with clear logic flow
- State key lemmas and theorems explicitly before using them
- Use numbered equations for important results

### Mathematical Rigor
- Show all algebraic manipulations explicitly
- Verify identities through direct calculation when possible
- Rather than using "it can be shown" or similar phrases, actually show the work
- Handle edge cases and special conditions carefully

### Notation and Formatting
- Use consistent mathematical notation throughout
- Prefer LaTeX formatting for mathematical expressions, but ideally do so in a markdown document
- Use clear variable definitions and maintain them consistently
- Include both symbolic and concrete examples when helpful

## AI Assistant Behavior Preferences

### Work Persistence
- When user says "complete them" or similar, work through all remaining details
- Don't give up on complex algebraic manipulations
- Show intermediate steps even if they're lengthy
- A calculation being lengthy isn't reason to give up; complete detailed calculations if they are promising
- If the current approach seems like a dead end, try alternative approaches rather than stopping

### Documentation Style
- Be concise in explanations but thorough in mathematical detail
- Focus on the "why" and "how" of mathematical reasoning
- Minimize conversational overhead, maximize mathematical content
- Use technical language appropriately without over-explaining

### File Management
- Create new proof attempts when previous ones are incomplete
- Use systematic naming (proof_attempt_[letter][number], etc.)
- Read and learn from previous attempts before starting new ones
- Build upon insights from earlier work, but don't cite results from previous proof attempts (re-prove them instead) unless the previous attempt been approved and finalized

### Problem-Solving Approach
- Start with structural insights before diving into algebra
- Use induction proofs systematically (base case, inductive step)
- When algebra gets complex, consider multiple approaches
- Verify results with specific examples when possible

## Communication Style
- Be direct and focused on the mathematical content
- Avoid unnecessary preambles or conclusions
- Let the mathematics speak for itself
- When user asks for completion, dive directly into the work

## Other
- When compiling LaTeX, remove aux and log files after successful compilation
- If you try to read a file and it fails either due to length or not being found or some issue, DO NOT JUST IGNORE IT, stop and ask what to do about it
- Never complain that something "would be" difficult and do something simpler; instead, state that it WILL BE difficult, and then DO IT FULLY.