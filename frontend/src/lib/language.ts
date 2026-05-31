export type LongVowelStyle = 'expand' | 'collapse' | 'keepMark' | 'remove';
export type KanaLongVowelStyle = 'keep' | 'expand' | 'collapse';

export interface KanaToHangulOptions {
	/**
	 * ん / ン을 다음 소리에 따라 ㄴ, ㅁ, ㅇ으로 동화할지 여부입니다.
	 *
	 * false:
	 * - しんぶん -> 신분
	 *
	 * true:
	 * - しんぶん -> 심분
	 * - まんが -> 망가
	 */
	assimilateNasal?: boolean;

	/**
	 * 조사 は, へ, を를 발음 기준으로 단순 치환합니다.
	 *
	 * - は -> わ
	 * - へ -> え
	 * - を -> お
	 *
	 * 단, 형태소 분석이 아니므로 완벽하지 않습니다.
	 */
	useSimpleParticleRules?: boolean;

	/**
	 * 매핑되지 않은 문자, 한자, 기호를 그대로 둘지 여부입니다.
	 */
	keepUnknown?: boolean;

	/**
	 * 장음 기호 ー 처리 방식입니다.
	 *
	 * expand:
	 * - スーパー -> 스우파아
	 * - チョコレート -> 초코레에토
	 *
	 * collapse:
	 * - スーパー -> 스파
	 * - チョコレート -> 초코레토
	 *
	 * keepMark:
	 * - スーパー -> 스ー파ー
	 *
	 * remove:
	 * - スーパー -> 스파
	 */
	longVowelStyle?: LongVowelStyle;

	/**
	 * 변환 결과의 반복 장모음을 UI용으로 자연스럽게 줄입니다.
	 *
	 * - 레에 -> 레
	 * - 스우 -> 스
	 * - 파아 -> 파
	 * - 토우 -> 토
	 *
	 * 주의:
	 * 실제로 분리된 음절도 줄어들 수 있으므로 UI 표시용일 때만 켜는 것을 권장합니다.
	 */
	naturalizeRepeatedVowels?: boolean;

	/**
	 * 히라가나로 적힌 장음 처리 방식입니다.
	 *
	 * 예:
	 * - ごとう: ご + と + う
	 * - とうきょう: と + う + きょ + う
	 * - せんせい: せ + ん + せ + い
	 *
	 * keep:
	 * - ごとう -> 고토우
	 *
	 * expand:
	 * - ごとう -> 고토오
	 *
	 * collapse:
	 * - ごとう -> 고토
	 */
	kanaLongVowelStyle?: KanaLongVowelStyle;
}

const DEFAULT_OPTIONS: Required<KanaToHangulOptions> = {
	assimilateNasal: true,
	useSimpleParticleRules: true,
	keepUnknown: true,
	longVowelStyle: 'expand',
	naturalizeRepeatedVowels: true,
	kanaLongVowelStyle: 'collapse'
};

type KanaVowel = 'a' | 'i' | 'u' | 'e' | 'o';

function getKanaVowel(kanaUnit: string): KanaVowel | null {
	const unit = katakanaToHiragana(kanaUnit);

	const last = unit[unit.length - 1];

	if ('あかがさざただなはばぱまやらわぁゃ'.includes(last)) return 'a';
	if ('いきぎしじちぢにひびぴみりゐぃ'.includes(last)) return 'i';
	if ('うくぐすずつづぬふぶぷむゆるぅゅゔ'.includes(last)) return 'u';
	if ('えけげせぜてでねへべぺめれゑぇ'.includes(last)) return 'e';
	if ('おこごそぞとどのほぼぽもよろをぉょ'.includes(last)) return 'o';

	return null;
}

/**
 * 히라가나로 적힌 장음인지 판단합니다.
 *
 * 대표 패턴:
 * - お段 + う: とう, こう, しょう, きょう
 * - お段 + お: おおさか
 * - え段 + い: せい, れい, けい
 * - え段 + え: ねえ
 */
function isKanaLongVowelFollower(previousVowel: KanaVowel | null, currentKana: string): boolean {
	if (!previousVowel) return false;

	const current = katakanaToHiragana(currentKana);

	if (previousVowel === 'o' && (current === 'う' || current === 'お')) {
		return true;
	}

	if (previousVowel === 'e' && (current === 'い' || current === 'え')) {
		return true;
	}

	return false;
}

function handleKanaLongVowel(result: string, style: KanaLongVowelStyle): string {
	switch (style) {
		case 'keep':
			return result;

		case 'expand':
			return appendLongVowel(result);

		case 'collapse':
			return result;

		default:
			return result;
	}
}

const CHOSEONG = [
	'ㄱ',
	'ㄲ',
	'ㄴ',
	'ㄷ',
	'ㄸ',
	'ㄹ',
	'ㅁ',
	'ㅂ',
	'ㅃ',
	'ㅅ',
	'ㅆ',
	'ㅇ',
	'ㅈ',
	'ㅉ',
	'ㅊ',
	'ㅋ',
	'ㅌ',
	'ㅍ',
	'ㅎ'
] as const;

const JUNGSEONG = [
	'ㅏ',
	'ㅐ',
	'ㅑ',
	'ㅒ',
	'ㅓ',
	'ㅔ',
	'ㅕ',
	'ㅖ',
	'ㅗ',
	'ㅘ',
	'ㅙ',
	'ㅚ',
	'ㅛ',
	'ㅜ',
	'ㅝ',
	'ㅞ',
	'ㅟ',
	'ㅠ',
	'ㅡ',
	'ㅢ',
	'ㅣ'
] as const;

const JONGSEONG = [
	'',
	'ㄱ',
	'ㄲ',
	'ㄳ',
	'ㄴ',
	'ㄵ',
	'ㄶ',
	'ㄷ',
	'ㄹ',
	'ㄺ',
	'ㄻ',
	'ㄼ',
	'ㄽ',
	'ㄾ',
	'ㄿ',
	'ㅀ',
	'ㅁ',
	'ㅂ',
	'ㅄ',
	'ㅅ',
	'ㅆ',
	'ㅇ',
	'ㅈ',
	'ㅊ',
	'ㅋ',
	'ㅌ',
	'ㅍ',
	'ㅎ'
] as const;

interface HangulParts {
	cho: string;
	jung: string;
	jong: string;
}

function composeHangul(cho: string, jung: string, jong = ''): string {
	const choIndex = CHOSEONG.indexOf(cho as (typeof CHOSEONG)[number]);
	const jungIndex = JUNGSEONG.indexOf(jung as (typeof JUNGSEONG)[number]);
	const jongIndex = JONGSEONG.indexOf(jong as (typeof JONGSEONG)[number]);

	if (choIndex === -1 || jungIndex === -1 || jongIndex === -1) {
		return cho + jung + jong;
	}

	return String.fromCharCode(0xac00 + choIndex * 21 * 28 + jungIndex * 28 + jongIndex);
}

function decomposeHangul(char: string): HangulParts | null {
	if (!char) return null;

	const code = char.charCodeAt(0);
	const base = code - 0xac00;

	if (base < 0 || base > 11171) {
		return null;
	}

	const choIndex = Math.floor(base / (21 * 28));
	const jungIndex = Math.floor((base % (21 * 28)) / 28);
	const jongIndex = base % 28;

	return {
		cho: CHOSEONG[choIndex],
		jung: JUNGSEONG[jungIndex],
		jong: JONGSEONG[jongIndex]
	};
}

function isHangulSyllable(char: string): boolean {
	return decomposeHangul(char) !== null;
}

function replaceJong(char: string, jong: string): string {
	const parts = decomposeHangul(char);
	if (!parts) return char;

	return composeHangul(parts.cho, parts.jung, jong);
}

function canAttachJong(char: string): boolean {
	const parts = decomposeHangul(char);
	return Boolean(parts && !parts.jong);
}

function attachJongToLastSyllable(result: string, jong: string): string {
	if (!result) return result + jong;

	const last = result[result.length - 1];

	if (!canAttachJong(last)) {
		return result + jong;
	}

	return result.slice(0, -1) + replaceJong(last, jong);
}

/**
 * Katakana -> Hiragana
 */
function katakanaToHiragana(input: string): string {
	return input.replace(/[\u30a1-\u30f6]/g, (char) => {
		return String.fromCharCode(char.charCodeAt(0) - 0x60);
	});
}

const KANA_MAP: Record<string, string> = {
	// vowels
	あ: '아',
	い: '이',
	う: '우',
	え: '에',
	お: '오',

	// k
	か: '카',
	き: '키',
	く: '쿠',
	け: '케',
	こ: '코',

	// s
	さ: '사',
	し: '시',
	す: '스',
	せ: '세',
	そ: '소',

	// t
	た: '타',
	ち: '치',
	つ: '쓰',
	て: '테',
	と: '토',

	// n
	な: '나',
	に: '니',
	ぬ: '누',
	ね: '네',
	の: '노',

	// h
	は: '하',
	ひ: '히',
	ふ: '후',
	へ: '헤',
	ほ: '호',

	// m
	ま: '마',
	み: '미',
	む: '무',
	め: '메',
	も: '모',

	// y
	や: '야',
	ゆ: '유',
	よ: '요',

	// r
	ら: '라',
	り: '리',
	る: '루',
	れ: '레',
	ろ: '로',

	// w
	わ: '와',
	ゐ: '이',
	ゑ: '에',
	を: '오',

	// voiced g
	が: '가',
	ぎ: '기',
	ぐ: '구',
	げ: '게',
	ご: '고',

	// voiced z
	ざ: '자',
	じ: '지',
	ず: '즈',
	ぜ: '제',
	ぞ: '조',

	// voiced d
	だ: '다',
	ぢ: '지',
	づ: '즈',
	で: '데',
	ど: '도',

	// voiced b
	ば: '바',
	び: '비',
	ぶ: '부',
	べ: '베',
	ぼ: '보',

	// p
	ぱ: '파',
	ぴ: '피',
	ぷ: '푸',
	ぺ: '페',
	ぽ: '포',

	// small vowels
	ぁ: '아',
	ぃ: '이',
	ぅ: '우',
	ぇ: '에',
	ぉ: '오',

	// small ya / yu / yo
	ゃ: '야',
	ゅ: '유',
	ょ: '요',

	// small wa
	ゎ: '와',

	// ヴ after normalization
	ゔ: '부'
};

const COMBO_MAP: Record<string, string> = {
	// k
	きゃ: '캬',
	きゅ: '큐',
	きょ: '쿄',
	きぇ: '켸',

	// g
	ぎゃ: '갸',
	ぎゅ: '규',
	ぎょ: '교',
	ぎぇ: '계',

	// s / sh
	しゃ: '샤',
	しゅ: '슈',
	しょ: '쇼',
	しぇ: '셰',

	// j
	じゃ: '자',
	じゅ: '주',
	じょ: '조',
	じぇ: '제',
	ぢゃ: '자',
	ぢゅ: '주',
	ぢょ: '조',
	ぢぇ: '제',

	// ch
	ちゃ: '차',
	ちゅ: '추',
	ちょ: '초',
	ちぇ: '체',

	// n
	にゃ: '냐',
	にゅ: '뉴',
	にょ: '뇨',
	にぇ: '녜',

	// h
	ひゃ: '햐',
	ひゅ: '휴',
	ひょ: '효',
	ひぇ: '혜',

	// b
	びゃ: '뱌',
	びゅ: '뷰',
	びょ: '뵤',
	びぇ: '볘',

	// p
	ぴゃ: '퍄',
	ぴゅ: '퓨',
	ぴょ: '표',
	ぴぇ: '폐',

	// m
	みゃ: '먀',
	みゅ: '뮤',
	みょ: '묘',
	みぇ: '몌',

	// r
	りゃ: '랴',
	りゅ: '류',
	りょ: '료',
	りぇ: '례',

	// f sounds
	ふぁ: '파',
	ふぃ: '피',
	ふぇ: '페',
	ふぉ: '포',
	ふゅ: '퓨',

	// v sounds
	ゔぁ: '바',
	ゔぃ: '비',
	ゔぇ: '베',
	ゔぉ: '보',

	// w sounds
	うぃ: '위',
	うぇ: '웨',
	うぉ: '워',

	// kw / gw
	くぁ: '콰',
	くぃ: '퀴',
	くぇ: '퀘',
	くぉ: '쿼',
	ぐぁ: '과',
	ぐぃ: '귀',
	ぐぇ: '궤',
	ぐぉ: '궈',

	// ts
	つぁ: '차',
	つぃ: '치',
	つぇ: '체',
	つぉ: '초',

	// t foreign
	てぃ: '티',
	てゅ: '튜',
	とぅ: '투',

	// d foreign
	でぃ: '디',
	でゅ: '듀',
	どぅ: '두'
};

const LONG_VOWEL_MAP: Record<string, string> = {
	ㅏ: '아',
	ㅐ: '애',
	ㅑ: '야',
	ㅒ: '얘',
	ㅓ: '어',
	ㅔ: '에',
	ㅕ: '여',
	ㅖ: '예',
	ㅗ: '오',
	ㅘ: '아',
	ㅙ: '애',
	ㅚ: '외',
	ㅛ: '요',
	ㅜ: '우',
	ㅝ: '어',
	ㅞ: '에',
	ㅟ: '위',
	ㅠ: '유',
	ㅡ: '으',
	ㅢ: '의',
	ㅣ: '이'
};

function getLastHangulChar(text: string): string | null {
	for (let i = text.length - 1; i >= 0; i--) {
		if (isHangulSyllable(text[i])) {
			return text[i];
		}
	}

	return null;
}

function appendLongVowel(result: string): string {
	const last = getLastHangulChar(result);
	if (!last) return result + 'ー';

	const parts = decomposeHangul(last);
	if (!parts) return result + 'ー';

	return result + (LONG_VOWEL_MAP[parts.jung] ?? '');
}

function handleLongVowelMark(result: string, style: LongVowelStyle): string {
	switch (style) {
		case 'expand':
			return appendLongVowel(result);

		case 'collapse':
			return result;

		case 'remove':
			return result;

		case 'keepMark':
			return result + 'ー';

		default:
			return result;
	}
}

function getSokuonJongByNextKana(nextKana: string | undefined): string {
	const next = katakanaToHiragana(nextKana ?? '');

	if ('かきくけこがぎぐげご'.includes(next)) return 'ㄱ';
	if ('さしすせそざじずぜぞ'.includes(next)) return 'ㅅ';
	if ('たちつてとだぢづでど'.includes(next)) return 'ㅅ';
	if ('はひふへほばびぶべぼぱぴぷぺぽ'.includes(next)) return 'ㅂ';

	return 'ㅅ';
}

function handleSokuon(result: string, nextKana: string | undefined): string {
	const jong = getSokuonJongByNextKana(nextKana);
	return attachJongToLastSyllable(result, jong);
}

function getNasalJongByNextKana(
	nextKana: string | undefined,
	options: Required<KanaToHangulOptions>
): string {
	if (!options.assimilateNasal) return 'ㄴ';

	const next = katakanaToHiragana(nextKana ?? '');

	if ('まみむめもばびぶべぼぱぴぷぺぽ'.includes(next)) return 'ㅁ';
	if ('かきくけこがぎぐげご'.includes(next)) return 'ㅇ';

	return 'ㄴ';
}

function handleNasal(
	result: string,
	nextKana: string | undefined,
	options: Required<KanaToHangulOptions>
): string {
	const jong = getNasalJongByNextKana(nextKana, options);
	return attachJongToLastSyllable(result, jong);
}

function applySimpleParticleRules(input: string): string {
	return input
		.replace(/(^|[\s、。！？!?])は(?=$|[\s、。！？!?])/g, '$1わ')
		.replace(/(^|[\s、。！？!?])へ(?=$|[\s、。！？!?])/g, '$1え')
		.replace(/を/g, 'お');
}

function normalizeKana(input: string, options: Required<KanaToHangulOptions>): string {
	let text = String(input ?? '');

	// Half-width katakana, compatibility forms normalize.
	text = text.normalize('NFKC');

	// Katakana to Hiragana.
	text = katakanaToHiragana(text);

	if (options.useSimpleParticleRules) {
		text = applySimpleParticleRules(text);
	}

	return text;
}

interface KanaMatch {
	value: string;
	length: number;
}

function hasOwnRecordValue(record: Record<string, string>, key: string): boolean {
	return Object.prototype.hasOwnProperty.call(record, key);
}

function findKanaMatch(text: string, index: number): KanaMatch | null {
	for (const len of [3, 2]) {
		const part = text.slice(index, index + len);

		if (hasOwnRecordValue(COMBO_MAP, part)) {
			return {
				value: COMBO_MAP[part],
				length: len
			};
		}
	}

	const char = text[index];

	if (hasOwnRecordValue(KANA_MAP, char)) {
		return {
			value: KANA_MAP[char],
			length: 1
		};
	}

	return null;
}

/**
 * UI용 장음 자연화입니다.
 *
 * 예:
 * - 레에 -> 레
 * - 스우 -> 스
 * - 파아 -> 파
 * - 코오 -> 코
 * - 쿄우 -> 쿄
 */
function naturalizeRepeatedVowels(text: string): string {
	let result = '';

	for (const char of text) {
		const prev = result[result.length - 1];
		const prevParts = prev ? decomposeHangul(prev) : null;

		if (prevParts && !prevParts.jong) {
			const expectedLongVowel = LONG_VOWEL_MAP[prevParts.jung];

			if (char === expectedLongVowel) {
				continue;
			}
		}

		result += char;
	}

	return result;
}

/**
 * Kana to Korean pronunciation.
 */
export function kanaToHangul(input: string, userOptions: KanaToHangulOptions = {}): string {
	const options: Required<KanaToHangulOptions> = {
		...DEFAULT_OPTIONS,
		...userOptions
	};

	const text = normalizeKana(input, options);

	let result = '';
	let previousKanaVowel: KanaVowel | null = null;

	for (let i = 0; i < text.length; i++) {
		const char = text[i];
		const nextChar = text[i + 1];

		// Small tsu: っ / ッ
		if (char === 'っ') {
			result = handleSokuon(result, nextChar);
			previousKanaVowel = null;
			continue;
		}

		// Nasal n: ん / ン
		if (char === 'ん') {
			result = handleNasal(result, nextChar, options);
			previousKanaVowel = null;
			continue;
		}

		// Long vowel mark: ー
		if (char === 'ー') {
			result = handleLongVowelMark(result, options.longVowelStyle);
			previousKanaVowel = null;
			continue;
		}

		/**
		 * 히라가나식 장음 처리:
		 *
		 * - と + う -> 토오 / 토
		 * - きょ + う -> 쿄오 / 쿄
		 * - せ + い -> 세에 / 세
		 */
		if (options.kanaLongVowelStyle !== 'keep' && isKanaLongVowelFollower(previousKanaVowel, char)) {
			result = handleKanaLongVowel(result, options.kanaLongVowelStyle);
			previousKanaVowel = null;
			continue;
		}

		const match = findKanaMatch(text, i);

		if (match) {
			result += match.value;

			const kanaUnit = text.slice(i, i + match.length);
			previousKanaVowel = getKanaVowel(kanaUnit);

			i += match.length - 1;
			continue;
		}

		if (options.keepUnknown) {
			result += char;
		}

		previousKanaVowel = null;
	}

	if (options.naturalizeRepeatedVowels) {
		result = naturalizeRepeatedVowels(result);
	}

	return result;
}

type Language = 'hangul' | 'hiragana';

export function detectLang(str: string): Language {
	const text = str.trim();

	if (text.length > 0 && /^[\p{Script=Hiragana}\s]+$/u.test(text)) {
		return 'hiragana';
	}

	return 'hangul';
}
