import asyncio
import os

from playwright.async_api import async_playwright


async def scrape_all_departments():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto("https://raumzeit.hka-iwi.de/timetables")

        title_value = await page.get_attribute("#form1\\:select_date_input", "title")
        if title_value and "Date selection" in title_value:
            date = "11/12/2025"  # MM/DD/YYYY
        else:  # Assume German
            date = "12.11.2025"  # DD.MM.YYYY

        # Replace the value directly
        await page.fill("#form1\\:select_date_input", date)

        # Trigger the "change" event so JS reacts to it
        await page.eval_on_selector("#form1\\:select_date_input", "el => el.dispatchEvent(new Event('change', { bubbles: true }))")

        # Click to open department dropdown
        await page.click("#form1\\:department_label")

        # Get all <li> options
        department_options = await page.query_selector_all("#form1\\:department_panel li")

        results = []
        for i, opt in enumerate(department_options):

            text = await opt.inner_text()
            if text == "Fakultäten" or text == "Faculties":
                continue  # skip this option
            elif text == "Einrichtungen" or text == "Departments":
                break  # stop processing options
            print(f"Option {i}: {text}")

            # Click the dropdown again each loop, because it closes after selection
            await page.click("#form1\\:department_label")
            department_options = await page.query_selector_all("#form1\\:department_panel li")

            # Select the i-th option
            await department_options[i].click()

            # Wait for timetable (or next dropdown, depending on workflow)
            await page.wait_for_timeout(1000)  # adjust if needed

            # Click to open the cost dropdown (should be "#form1\\:cost_label")
            await page.click("#form1\\:cost_label")
            # Get all <li> options for cost (should be "#form1\\:cost_panel li")
            cost_options = await page.query_selector_all("#form1\\:cost_panel li")
            print(f"Found {len(cost_options)} cost options")
            for j, cost_opt in enumerate(cost_options):
                cost_text = await cost_opt.inner_text()
                print(f"  Cost Option {j}: {cost_text}")

                # Click the dropdown again each loop, because it closes after selection
                await page.click("#form1\\:cost_label")
                cost_options = await page.query_selector_all("#form1\\:cost_panel li")

                # Select the j-th option
                await cost_options[j].click()

                # Wait for timetable (or next dropdown, depending on workflow)
                await page.wait_for_timeout(1000)  # adjust if needed
                html = await page.content()

                # Check if the course semester dropdown exists and is enabled
                course_semester_label = await page.query_selector("#form1\\:course_semester_label")
                if course_semester_label:
                    is_disabled = await course_semester_label.get_attribute("aria-disabled")
                    if not is_disabled or is_disabled == "false":
                        await page.click("#form1\\:course_semester_label")
                        # Get all <li> options for course semester
                        course_semester_options = await page.query_selector_all("#form1\\:course_semester_panel li")
                        print(f"    Found {len(course_semester_options)} course semester options")
                        for k, course_semester_opt in enumerate(course_semester_options):
                            course_semester_text = await course_semester_opt.inner_text()
                            print(f"      Course Semester Option {k}: {course_semester_text}")
                            # Click the dropdown again each loop, because it closes after selection
                            await page.click("#form1\\:course_semester_label")
                            course_semester_options = await page.query_selector_all("#form1\\:course_semester_panel li")
                            await course_semester_options[k].click()
                            await page.wait_for_timeout(1000)  # adjust if needed
                            html = await page.content()
                            # Save the HTML content to a file
                            folder_name = "data/timetables_html"
                            os.makedirs(folder_name, exist_ok=True)
                            filename = f"{text}-{cost_text}-{course_semester_text}.html".replace(" ", "_").replace("/", "_")
                            save_name = os.path.join(folder_name, filename)
                            with open(save_name, "w", encoding="utf-8") as f:
                                f.write(html)
                            results.append((text, cost_text, course_semester_text))
                    else:
                        print("    Course semester dropdown is disabled, skipping.")
                else:
                    print("    Course semester dropdown not found, skipping.")

        await browser.close()
        return results


if __name__ == "__main__":
    data = asyncio.run(scrape_all_departments())
    print("Scraped", len(data), "departments")
