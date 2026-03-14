"""Text dataset for transformer training.

A small bundled text corpus with character-level tokenization
for next-character prediction.
"""

# Four Shakespeare sonnets: 18, 29, 73, 130
SHAKESPEARE_SONNETS = """Shall I compare thee to a summer's day?
Thou art more lovely and more temperate.
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date.
Sometime too hot the eye of heaven shines,
And often is his gold complexion dimmed;
And every fair from fair sometime declines,
By chance, or nature's changing course, untrimmed;
But thy eternal summer shall not fade,
Nor lose possession of that fair thou ow'st;
Nor shall death brag thou wand'rest in his shade,
When in eternal lines to Time thou grow'st.
So long as men can breathe, or eyes can see,
So long lives this, and this gives life to thee.

When, in disgrace with fortune and men's eyes,
I all alone beweep my outcast state,
And trouble deaf heaven with my bootless cries,
And look upon myself and curse my fate,
Wishing me like to one more rich in hope,
Featured like him, like him with friends possessed,
Desiring this man's art and that man's scope,
With what I most enjoy contented least;
Yet in these thoughts myself almost despising,
Haply I think on thee, and then my state,
Like to the lark at break of day arising
From sullen earth, sings hymns at heaven's gate;
For thy sweet love remembered such wealth brings
That then I scorn to change my state with kings.

That time of year thou mayst in me behold
When yellow leaves, or none, or few, do hang
Upon those boughs which shake against the cold,
Bare ruined choirs, where late the sweet birds sang.
In me thou see'st the twilight of such day
As after sunset fadeth in the west,
Which by and by black night doth take away,
Death's second self, that seals up each day's rest.
In me thou see'st the glowing of such fire
That on the ashes of his youth doth lie,
As the death-bed whereon it must expire,
Consumed with that which it was nourished by.
This thou perceiv'st, which makes thy love more strong,
To love that well which thou must leave ere long.

My mistress' eyes are nothing like the sun;
Coral is far more red than her lips' red;
If snow be white, why then her breasts are dun;
If hairs be wires, wires grow on her head.
I have seen roses damasked, red and white,
But no such roses see I in her cheeks;
And in some perfumes is there more delight
Than in the breath that from my mistress reeks.
I love to hear her speak, yet I well know
That music hath a far more pleasing sound;
I grant I never saw a goddess go;
My mistress, when she walks, treads on the ground.
And yet, by heaven, I think my love as rare
As any she belied with false compare."""


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    """Build character-level vocabulary from text.

    Returns (char_to_id, id_to_char) mappings.
    Characters are sorted for deterministic ordering.
    """
    chars = sorted(set(text))
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_id, id_to_char


def prepare_sequences(
    text: str, char_to_id: dict[str, int], seq_len: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Create training sequences using a sliding window.

    For each window of seq_len characters, the input is the first
    seq_len characters and the target is the next seq_len characters
    (shifted by one position).

    Returns (inputs, targets) where each is a list of token ID sequences.
    """
    token_ids = [char_to_id[ch] for ch in text]
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for i in range(len(token_ids) - seq_len):
        inputs.append(token_ids[i:i + seq_len])
        targets.append(token_ids[i + 1:i + seq_len + 1])
    return inputs, targets
